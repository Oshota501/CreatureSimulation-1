"""
PPO で魚を学習させる（GPU 上で数千の水槽を並列シミュレーション）

    python train.py                       # 既定の設定で学習
    python train.py --total-steps 50e6    # もっと長く
    python train.py --resume              # 保存済みのチェックポイントから続きを学習

別のターミナルで `python play.py --watch` を動かすと、学習中の魚をライブで見られます。
"""
import argparse
import os
import time

import torch

from fish_env import FishConfig, FishEnv
from model import ActorCritic, load_checkpoint, save_checkpoint


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--rollout", type=int, default=64, help="1回の更新で各水槽を何ステップ進めるか")
    p.add_argument("--total-steps", type=float, default=30e6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--approach-coef", type=float, default=0.0,
                   help="餌に近づいたときの補助報酬（0 なら『食べたら報酬』のみ）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="checkpoints/fish_ppo.pt")
    p.add_argument("--save-every", type=int, default=10, help="何回の更新ごとに保存するか")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    global_step = 0
    if args.resume and os.path.exists(args.save):
        model, cfg, data = load_checkpoint(args.save, device)
        model.train()
        global_step = data["global_step"]
        print(f"resume: {args.save} ({global_step:,} steps)")
    else:
        cfg = FishConfig(approach_coef=args.approach_coef)
        model = None

    env = FishEnv(args.num_envs, cfg, device)
    if model is None:
        model = ActorCritic(env.obs_dim, args.hidden).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    N, T, D = args.num_envs, args.rollout, env.obs_dim
    batch = N * T
    mb_size = batch // args.minibatches
    num_updates = int(args.total_steps // batch)

    buf_obs = torch.zeros(T, N, D, device=device)
    buf_act = torch.zeros(T, N, 3, dtype=torch.long, device=device)
    buf_logp = torch.zeros(T, N, device=device)
    buf_rew = torch.zeros(T, N, device=device)
    buf_done = torch.zeros(T, N, device=device)
    buf_val = torch.zeros(T, N, device=device)

    print(f"device={device}  envs={N}  batch={batch:,}  updates={num_updates}  obs_dim={D}")
    obs = env.reset()
    t_start = time.time()

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates
        for g in optim.param_groups:
            g["lr"] = args.lr * frac

        # ---------------- データ収集（すべて GPU 上） ----------------
        model.eval()
        env.reset_stats()
        with torch.no_grad():
            for t in range(T):
                action, logp, value = model.act(obs)
                next_obs, reward, done, terminal_obs = env.step(action)
                # 時間切れは「失敗」ではないので、打ち切った先の価値を報酬に足しておく
                reward = reward + args.gamma * model.value(terminal_obs) * done
                buf_obs[t], buf_act[t], buf_logp[t] = obs, action, logp
                buf_rew[t], buf_done[t], buf_val[t] = reward, done.float(), value
                obs = next_obs

            # GAE
            next_val = model.value(obs)
            adv = torch.zeros(T, N, device=device)
            last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nv = next_val if t == T - 1 else buf_val[t + 1]
                nonterm = 1.0 - buf_done[t]
                delta = buf_rew[t] + args.gamma * nv * nonterm - buf_val[t]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[t] = last
            ret = adv + buf_val

        # ---------------- PPO 更新 ----------------
        model.train()
        b_obs, b_act = buf_obs.reshape(batch, D), buf_act.reshape(batch, 3)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for i in range(0, batch, mb_size):
                idx = perm[i : i + mb_size]
                new_logp, entropy, value = model.evaluate(b_obs[idx], b_act[idx])
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (new_logp - b_logp[idx]).exp()
                pg_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                v_loss = 0.5 * (value - b_ret[idx]).square().mean()
                ent = entropy.mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optim.step()

        global_step += batch

        # ---------------- ログ ----------------
        s = {k: v.item() for k, v in env.stats.items()}
        sps = update * batch / (time.time() - t_start)
        ep = f"{s['ep_return'] / s['ep_count']:6.2f}" if s["ep_count"] else "   -  "
        print(
            f"upd {update:4d}/{num_updates}  step {global_step / 1e6:6.2f}M  {sps / 1e3:5.1f}k sps  "
            f"餌/1000step {s['eaten'] / s['steps'] * 1000:6.2f}  ep_return {ep}  "
            f"speed {s['speed'] / s['steps']:.3f}  wall {s['wall'] / s['steps']:.3f}  "
            f"ent {ent.item():.2f}  v_loss {v_loss.item():.4f}",
            flush=True,
        )
        if update % args.save_every == 0 or update == num_updates:
            save_checkpoint(args.save, model, cfg, model.actor[0].out_features, global_step)

    print(f"done. saved to {args.save}")


if __name__ == "__main__":
    main()
