"""
生命体の進化シミュレーション（GPU 上で全個体・全餌をまとめて計算）

- 各個体は遺伝子（体の形・ヒレの配置・顎や鎧・食性・生き方・脳の重み）を持つ
- 生きているだけ・泳ぐだけでエネルギーが減り、植物や肉を食べると増える
- エネルギーが十分たまったら子を産む。子は親の遺伝子を少しだけ突然変異させて受け継ぐ
- エネルギーが尽きる・体力が尽きる（噛まれて殺される）・寿命が来ると死に、体重に応じた肉（死体）が残る
- 個体数が減っても自動では補充しない（表示画面でマウスから生き物を置ける）

体のつくり
- 尾びれ: 左右に振ると前に進む（生まれつきの遊泳リズム＋脳で制御）
- ヒレ: 最大 3 対。付く位置・向き・大きさは遺伝子で決まり、脳が 1 枚ずつ動かす。
  ヒレは向いている方向へ水を押すので、体は逆向きに押される。
  例) 顔に前向きのヒレ → 使うとバック / 後ろ向きのヒレ → 前進 / 前のほうの横向きのヒレ → 旋回
- 頭（口）: 相手の体に口が触れているときに脳が「噛む」を選ぶと噛みつく。顎が強く頭と体が大きいほど深く噛める
- 鎧: 受けるダメージを減らすが、重くなり維持にエネルギーを使う
- 食性: 0 に近いほど植物を、1 に近いほど肉を効率よく消化できる（どっちつかずは損）

座標系は fish_env.py と同じ（x 右、y 上、θ が増えると左に曲がる）。
世界の端はつながっている（右端から出ると左端から入る）。
"""
import math
import os
from dataclasses import asdict, dataclass

import torch

N_FIN_PAIRS = 3

# 遺伝子: (名前, 最小, 最大)。遺伝子の生の値は実数で、sigmoid で範囲に写す
TRAITS = [
    # 体の形
    ("size", 0.6, 1.8),           # 全体の大きさ: 大きいほど速く口も大きいが、燃費が悪い
    ("body_len", 0.6, 1.6),       # 体の長さ: 長いほど水の抵抗が小さいが、曲がりにくい
    ("body_wid", 0.5, 1.6),       # 体の太さ: 太いほどエネルギーをためられるが、重く抵抗も大きい
    ("bulk_pos", 0.2, 0.7),       # 一番太い位置（0=頭寄り, 1=尾寄り）
    ("tail_power", 0.3, 2.0),     # 尾びれの大きさ: 推進力が上がるが、泳ぐエネルギーも増える
    ("tail_fork", 0.0, 1.0),      # 尾びれの切れ込み（見た目だけ）
    # 捕食
    ("head_size", 0.6, 1.6),      # 頭の大きさ: 口が大きく噛む力も強いが、重い
    ("jaw", 0.0, 1.5),            # 顎（攻撃力）: 強いほど深く噛めるが、維持にエネルギーを使う
    ("armor", 0.0, 1.0),          # 鎧（トゲ）: 受けるダメージを減らすが、重く燃費が悪い
    ("diet", 0.0, 1.0),           # 食性: 0=植物をよく消化（草食）… 1=肉をよく消化（肉食）
    # センサー（目）
    ("fov", 40.0, 320.0),         # 視野角（度）。広いほど目が横に付く
    ("sense_range", 10.0, 70.0),  # 見える距離。遠いほど目が大きい。広く遠いほど維持コストが高い
    # 泳ぎのリズム
    ("cpg_freq", 0.02, 0.2),      # 生まれつきの遊泳リズムの速さ（1ステップあたりの周期数）
    ("cpg_amp", 0.0, 3.0),        # 遊泳リズムが尻尾を動かす強さ
    # 生き方
    ("repro_at", 0.4, 0.95),      # エネルギーが最大の何割たまったら子を産むか
    ("child_share", 0.15, 0.6),   # 子に渡すエネルギー（最大エネルギーに対する割合）
    ("mutation", 0.005, 0.12),    # 子に伝わるときの突然変異の強さ（これ自体も進化する）
    # 模様（見た目だけ。少しずつずれていくので、家族ごとに見た目が分かれていく）
    ("stripes", 0.0, 4.99),       # 縞の本数
    ("pattern_hue", 0.0, 1.0),    # 縞とヒレの色（体色からのずれ）
]
for _i in range(N_FIN_PAIRS):
    TRAITS += [
        (f"fin{_i}_size", -0.8, 1.6),   # 0 以下ならこのヒレは生えていない
        (f"fin{_i}_pos", 0.05, 0.95),   # 付いている位置（0=顔, 1=尾の付け根）
        (f"fin{_i}_angle", 0.0, 180.0), # 向き（0=前向き→使うとバック, 90=横向き, 180=後ろ向き→使うと前進）
    ]
N_TRAITS = len(TRAITS)
T = {name: i for i, (name, _, _) in enumerate(TRAITS)}
FIN_SIZE = [T[f"fin{i}_size"] for i in range(N_FIN_PAIRS)]
FIN_POS = [T[f"fin{i}_pos"] for i in range(N_FIN_PAIRS)]
FIN_ANGLE = [T[f"fin{i}_angle"] for i in range(N_FIN_PAIRS)]
# 最初の個体は顎と鎧が弱めの状態から始める（遺伝子の生の値の平均）
INIT_BIAS = {"jaw": -1.0, "armor": -1.5}
HUE = N_TRAITS        # 体色。生存には関係しない目印で、系統ごとに少しずつずれていく
N_MORPH = N_TRAITS + 1
N_OUT = 1 + 2 * N_FIN_PAIRS + 1   # 脳の出力: 尻尾, 左ヒレ×3, 右ヒレ×3, 噛む
BITE_OUT = N_OUT - 1
SAVE_VERSION = 3


def pair_sense(head, cos_t, sin_t, fov, rng, targets, target_ok, world_size, n_sectors: int):
    """全個体 × 全対象の距離を測り、視野を n_sectors 個の扇形に分けて扇形ごとに一番近いものの近さ (0..1) を返す。
    世界の端はつながっている。角度は + が左。"""
    rel = targets[None, :, :] - head[:, None, :]
    rel = torch.remainder(rel + world_size / 2, world_size) - world_size / 2
    fwd = rel[..., 0] * cos_t[:, None] + rel[..., 1] * sin_t[:, None]
    lat = -rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
    dist = torch.where(target_ok[None, :], torch.sqrt(fwd * fwd + lat * lat), torch.inf)
    ang = torch.atan2(lat, fwd)
    half = (fov / 2)[:, None]
    vis = (ang.abs() < half) & (dist < rng[:, None])
    sec = ((ang + half) / fov[:, None] * n_sectors).floor().clamp(0, n_sectors - 1)
    close = torch.where(vis, 1 - dist / rng[:, None], 0.0)
    sense = torch.stack([torch.where(sec == s, close, 0.0).amax(1) for s in range(n_sectors)], 1)
    return sense, dist


def eat_and_sense(head, cos_t, sin_t, fov, rng, pos, ok, alive, reach, radius, world_size, n_sectors: int):
    """口が触れているもの（植物・肉）を決め（1つにつき一番近い1匹だけ）、同時にセンサーの値を返す"""
    sense, dist = pair_sense(head, cos_t, sin_t, fov, rng, pos, ok, world_size, n_sectors)
    cand = torch.where((dist < reach[:, None] + radius[None, :]) & alive[:, None], dist, torch.inf)
    best, eater = cand.min(0)
    return sense, torch.isfinite(best), eater


def see_others(head, cos_t, sin_t, fov, rng, pos, alive, world_size, n_sectors: int):
    """他の個体を見る: 扇形ごとに一番近い個体の近さ (0..1) と、その個体の番号を返す"""
    C = head.shape[0]
    idx = torch.arange(C, device=head.device)
    rel = pos[None, :, :] - head[:, None, :]
    rel = torch.remainder(rel + world_size / 2, world_size) - world_size / 2
    fwd = rel[..., 0] * cos_t[:, None] + rel[..., 1] * sin_t[:, None]
    lat = -rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
    ok = alive[None, :] & (idx[:, None] != idx[None, :])
    dist = torch.where(ok, torch.sqrt(fwd * fwd + lat * lat), torch.inf)
    ang = torch.atan2(lat, fwd)
    half = (fov / 2)[:, None]
    vis = (ang.abs() < half) & (dist < rng[:, None])
    sec = ((ang + half) / fov[:, None] * n_sectors).floor().clamp(0, n_sectors - 1)
    close = torch.where(vis, 1 - dist / rng[:, None], 0.0)
    near, who = [], []
    for s in range(n_sectors):
        m, k = torch.where(sec == s, close, 0.0).max(1)
        near.append(m)
        who.append(k)
    return torch.stack(near, 1), torch.stack(who, 1)


def find_bites(head, cos_t, sin_t, pos, alive, seg_front, seg_back, half_w, mouth, want_bite, world_size):
    """噛みつける相手を探す: 自分の口（頭の先）が相手の体（頭から尾までの細長い形）に触れていれば噛める"""
    C = head.shape[0]
    idx = torch.arange(C, device=head.device)
    rel = pos[None, :, :] - head[:, None, :]
    rel = torch.remainder(rel + world_size / 2, world_size) - world_size / 2
    # 相手の体の座標系で、自分の頭から相手の背骨（線分）までの距離
    cj, sj = cos_t[None, :], sin_t[None, :]
    along = -(rel[..., 0] * cj + rel[..., 1] * sj)
    side = rel[..., 0] * sj - rel[..., 1] * cj
    clamped = torch.minimum(torch.maximum(along, -seg_back[None, :]), seg_front[None, :])
    gap = torch.sqrt((along - clamped) ** 2 + side * side) - half_w[None, :]
    ok = alive[None, :] & (idx[:, None] != idx[None, :]) & want_bite[:, None]
    best, victim = torch.where(ok, gap, torch.inf).min(1)
    return best < mouth, victim


def _compiled(fn):
    """torch.compile で GPU カーネルをまとめて高速化（使えない環境ではそのまま実行）"""
    state = {"fn": None}

    def run(*args):
        if state["fn"] is None:
            try:
                state["fn"] = torch.compile(fn)
                return state["fn"](*args)
            except Exception as e:  # noqa: BLE001
                print(f"[evo_world] torch.compile が使えないので通常実行します: {type(e).__name__}")
                state["fn"] = fn
        return state["fn"](*args)

    return run


eat_and_sense_fast = _compiled(eat_and_sense)
see_others_fast = _compiled(see_others)
find_bites_fast = _compiled(find_bites)


def body_width_at(t, width_max, bulk_pos):
    """頭(t=0)から尾の付け根(t=1)までの体の太さ（片側）。bulk_pos で一番太い位置が決まる"""
    p = math.log(0.5) / torch.log(bulk_pos)
    return width_max * (0.18 + 0.82 * torch.sin(math.pi * t.clamp(1e-4, 1) ** p))


@dataclass
class EvoConfig:
    world_w: float = 300.0
    world_h: float = 240.0
    max_creatures: int = 2048      # 同時に存在できる最大数（GPU 上の枠の数）
    init_creatures: int = 30
    max_births_per_step: int = 64
    # 植物
    max_food: int = 2000           # 植物の枠の数（手で撒ける上限）
    plant_capacity: int = 1200     # 自然に生える上限。最初はここまで生えている（エデンの園）
    food_energy: float = 25.0
    food_radius: float = 0.8
    food_regrow: float = 0.003     # 生える速さ: 1ステップに (上限 - 今の数) × これ だけ生える
    n_hotspots: int = 3            # 植物がよく生える場所（最初の数。あとから置いたり消したりできる）
    hotspot_sigma: float = 24.0    # よく生える場所の広さ（最初の値）
    hotspot_frac: float = 0.7      # 生える植物のうち、よく生える場所に生える割合
    # 肉（死体）
    max_meat: int = 512
    meat_per_mass: float = 60.0    # 死体のエネルギー = これ × 体重（大きい個体ほどごちそう）
    meat_decay: float = 0.001      # 死体は少しずつ腐って小さくなる（1ステップあたりの割合）
    meat_bite: float = 10.0        # 1回にかじり取れる肉の量（× 大きさ² × (0.5 + 顎) × 頭の大きさ）
    # 噛みつき・体力
    bite_power: float = 4.0        # 噛みつきのダメージ（× 顎 × 大きさ² × 頭の大きさ）
    blood_frac: float = 0.4        # ダメージのうち、噛まれた側から噛んだ側へ移るエネルギーの割合
    hp_per_mass: float = 50.0      # 最大体力 = これ × 体重
    hp_regen: float = 0.002        # 1ステップで回復する体力（最大体力に対する割合）
    armor_block: float = 0.75      # 鎧が最大のとき、受けるダメージを何割減らすか
    diet_specialization: float = 1.0  # 食性の専門化の強さ（1 より大きいほど「どっちつかず」が不利）
    # センサーと脳
    n_sectors: int = 7
    hidden: int = 16
    # 尾びれ（fish_env.py と同じモデル）
    tail_accel: float = 0.12
    tail_damp: float = 0.10
    tail_spring: float = 0.02
    tail_vmax: float = 0.45
    tail_max_deg: float = 50.0
    thrust_coef: float = 0.30
    tail_yaw_coef: float = 0.03
    # ヒレ
    fin_force: float = 0.006       # ヒレ1枚（大きさ1）を全力で使ったときの力
    fin_rot_gain: float = 6.0      # ヒレの力が体を回す効きやすさ
    fin_idle_drag: float = 0.004   # ヒレがあるだけで増える水の抵抗（大きさ1あたり）
    # 水の抵抗
    drag_forward: float = 0.05
    drag_lateral: float = 0.35
    drag_angular: float = 0.25
    # エネルギー
    energy_per_mass: float = 100.0  # 最大エネルギー = これ × 体重
    basal_cost: float = 0.02        # 生きているだけでかかる（× 体重）
    move_cost: float = 0.5          # 尻尾を振るとかかる（× 尾びれの大きさ × 大きさ² × 振り²）
    fin_use_cost: float = 0.01      # ヒレを使うとかかる（× 使った強さ × ヒレの大きさ × 大きさ²）
    fin_keep_cost: float = 0.002    # ヒレを持っているだけでかかる（× ヒレの大きさ × 大きさ²）
    jaw_keep_cost: float = 0.004    # 顎を持っているだけでかかる（× 顎 × 大きさ²）
    armor_keep_cost: float = 0.006  # 鎧を持っているだけでかかる（× 鎧 × 体重）
    bite_cost: float = 0.02         # 噛みつこうとするとかかる（× 顎 × 大きさ²）
    sensor_cost: float = 0.03
    birth_overhead: float = 0.1     # 出産時に余分にかかるエネルギーの割合
    min_repro_age: int = 150
    max_age: int = 6000


class World:
    def __init__(self, cfg: EvoConfig | None = None, device="cpu"):
        self.cfg = cfg or EvoConfig()
        self.device = torch.device(device)
        c, dev = self.cfg, self.device
        C, M, MM, S, H = c.max_creatures, c.max_food, c.max_meat, c.n_sectors, c.hidden

        self.n_in = 5 * S + 9
        layout = [("w_in", (self.n_in, H)), ("w_h", (H, H)), ("b_h", (H,)), ("w_out", (H, N_OUT)), ("b_out", (N_OUT,))]
        self.brain_layout, off = {}, N_MORPH
        for name, shape in layout:
            n = math.prod(shape)
            self.brain_layout[name] = (off, off + n, shape)
            off += n
        self.n_genes = off
        self.lo = torch.tensor([t[1] for t in TRAITS], device=dev)
        self.hi = torch.tensor([t[2] for t in TRAITS], device=dev)
        self.init_bias = torch.zeros(N_TRAITS, device=dev)
        for name, b in INIT_BIAS.items():
            self.init_bias[T[name]] = b
        self.world_size = torch.tensor([c.world_w, c.world_h], device=dev)
        self.arange_c = torch.arange(C, device=dev)
        # ヒレの遺伝子の列番号（Python のリストで添字すると毎回 GPU へ転送が起きるので、先に GPU に置く）
        self.fin_cols = {k: torch.tensor(v, device=dev) for k, v in
                         (("size", FIN_SIZE), ("pos", FIN_POS), ("angle", FIN_ANGLE))}
        self.food_r = torch.full((M,), c.food_radius, device=dev)

        zf = lambda *s: torch.zeros(*s, device=dev)
        zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        # 個体
        self.alive = torch.zeros(C, dtype=torch.bool, device=dev)
        self.genes = zf(C, self.n_genes)
        self.pos, self.vel = zf(C, 2), zf(C, 2)
        self.theta, self.omega = zf(C), zf(C)
        self.tail, self.tail_vel = zf(C), zf(C)
        self.fin_act, self.bite_act = zf(C, 2 * N_FIN_PAIRS), zf(C)
        self.phase, self.hidden = zf(C), zf(C, H)
        self.energy, self.hp, self.hurt, self.age = zf(C), zf(C), zf(C), zl(C)
        self.generation, self.food_eaten, self.children, self.kills = zl(C), zf(C), zl(C), zl(C)
        self.cid, self.root, self.parent = zl(C), zl(C), zl(C)
        self.last_in, self.last_out = zf(C, self.n_in), zf(C, N_OUT)
        self.next_id = torch.zeros((), dtype=torch.long, device=dev)
        # 植物
        self.hotspots = torch.rand(c.n_hotspots, 2, device=dev) * self.world_size
        self.hotspot_r = torch.full((c.n_hotspots,), c.hotspot_sigma, device=dev)
        self.food_pos = self._sample_food_pos(M)
        self.food_alive = torch.arange(M, device=dev) < c.plant_capacity
        # 肉（死体）
        self.meat_pos, self.meat_energy = zf(MM, 2), zf(MM)
        self.meat_alive = torch.zeros(MM, dtype=torch.bool, device=dev)

        self.steps = 0
        self.totals = {k: torch.zeros((), device=dev) for k in ("births", "deaths", "eaten", "kills", "spawned")}

        n = c.init_creatures
        genes = self.random_genes(n)
        bd = self.body(self.traits(genes))
        minus1 = torch.full((n,), -1, device=dev, dtype=torch.long)
        self._spawn(self.arange_c[:n], torch.ones(n, dtype=torch.bool, device=dev), genes,
                    pos=torch.rand(n, 2, device=dev) * self.world_size,
                    theta=(torch.rand(n, device=dev) * 2 - 1) * math.pi,
                    energy=0.6 * bd["emax"], hp=bd["hp_max"], generation=zl(n), root=minus1, parent=minus1)

    # ================================================================ 遺伝子
    def traits(self, genes=None):
        g = self.genes if genes is None else genes
        return self.lo + (self.hi - self.lo) * torch.sigmoid(g[:, :N_TRAITS])

    def body(self, tr):
        """遺伝子から体の寸法・体重などを計算する"""
        c = self.cfg
        size, length, width = tr[:, T["size"]], tr[:, T["body_len"]], tr[:, T["body_wid"]]
        head, armor = tr[:, T["head_size"]], tr[:, T["armor"]]
        flesh = size.square() * length * width * (0.8 + 0.2 * head)   # 食べられる部分の重さ
        body_len = 6.0 * size * length
        return dict(size=size, length=length, width=width, head=head, flesh=flesh,
                    mass=flesh * (1 + 0.4 * armor),                    # 鎧のぶん重くなる
                    body_len=body_len, width_max=1.05 * size * width, nose=0.43 * body_len,
                    mouth=1.1 * size * (0.5 * head + 0.5 * width.sqrt()),
                    emax=c.energy_per_mass * flesh, hp_max=c.hp_per_mass * flesh)

    def random_genes(self, n):
        g = torch.zeros(n, self.n_genes, device=self.device)
        g[:, :N_TRAITS] = torch.randn(n, N_TRAITS, device=self.device) + self.init_bias
        g[:, HUE] = torch.rand(n, device=self.device)
        for name, (a, b, shape) in self.brain_layout.items():
            if name.startswith("w"):
                g[:, a:b] = torch.randn(n, b - a, device=self.device) * (1.5 / math.sqrt(shape[0]))
            else:
                g[:, a:b] = torch.randn(n, b - a, device=self.device) * 0.3
        return g

    def mutate(self, genes):
        sigma = self.traits(genes)[:, T["mutation"]]
        noise = torch.randn_like(genes) * sigma[:, None]
        # 突然変異の強さ自体と体色は、決まった幅でゆっくり変化させる
        noise[:, T["mutation"]] = torch.randn(genes.shape[0], device=self.device) * 0.15
        noise[:, HUE] = torch.randn(genes.shape[0], device=self.device) * 0.015
        return genes + noise

    def _brain(self, name):
        a, b, shape = self.brain_layout[name]
        return self.genes[:, a:b].reshape(-1, *shape)

    # ================================================================ 誕生
    def _spawn(self, slots, valid, genes, pos, theta, energy, hp, generation, root, parent):
        """slots（重複なし）のうち valid な枠に新しい個体を書き込む"""
        ids = self.next_id + torch.cumsum(valid.long(), 0) - 1
        self.next_id = self.next_id + valid.sum()
        n = slots.shape[0]
        zf = lambda *s: torch.zeros(*s, device=self.device)
        zeros_l = torch.zeros_like(generation)
        values = dict(
            genes=genes, pos=pos, vel=zf(n, 2), theta=theta, omega=zf(n), tail=zf(n), tail_vel=zf(n),
            fin_act=zf(n, 2 * N_FIN_PAIRS), bite_act=zf(n), phase=torch.rand(n, device=self.device) * 2 * math.pi,
            hidden=zf(n, self.cfg.hidden), energy=energy, hp=hp, hurt=zf(n), age=zeros_l, generation=generation,
            food_eaten=zf(n), children=zeros_l, kills=zeros_l, cid=ids,
            root=torch.where(root < 0, ids, root), parent=parent,
            alive=torch.ones(n, dtype=torch.bool, device=self.device),
        )
        for name, v in values.items():
            cur = getattr(self, name)
            old = cur[slots]
            m = valid.view(-1, *([1] * (old.dim() - 1)))
            cur[slots] = torch.where(m, v.to(cur.dtype), old)

    def _pair_up(self, want, free, limit):
        """want な枠と free な枠を先頭から k 番目どうし組にする（GPU 上で完結、同期なし）"""
        limit = min(limit, len(want), len(free))
        a = torch.argsort((~want).long() * len(want) + torch.arange(len(want), device=self.device))[:limit]
        b = torch.argsort((~free).long() * len(free) + torch.arange(len(free), device=self.device))[:limit]
        k = torch.arange(limit, device=self.device)
        return a, b, (k < want.sum()) & (k < free.sum())

    def _births(self, tr, bd):
        c = self.cfg
        emax = bd["emax"]
        ready = self.alive & (self.energy > tr[:, T["repro_at"]] * emax) & (self.age > c.min_repro_age)
        parents, slots, valid = self._pair_up(ready, ~self.alive, min(c.max_births_per_step, c.max_creatures))
        B = len(parents)

        genes = self.mutate(self.genes[parents])
        child = self.body(self.traits(genes))
        e_parent = self.energy[parents]
        e_child = torch.minimum(tr[parents, T["child_share"]] * emax[parents], 0.8 * e_parent / (1 + c.birth_overhead))
        e_child = torch.minimum(e_child, child["emax"])
        self.energy[parents] = e_parent - torch.where(valid, e_child * (1 + c.birth_overhead), 0.0)
        self.children[parents] = self.children[parents] + valid.long()

        th = self.theta[parents]
        back = torch.stack([torch.cos(th), torch.sin(th)], -1) * (0.7 * bd["body_len"][parents])[:, None]
        pos = torch.remainder(self.pos[parents] - back + torch.randn(B, 2, device=self.device), self.world_size)
        self._spawn(slots, valid, genes, pos, th + torch.randn(B, device=self.device) * 0.5, e_child,
                    child["hp_max"], self.generation[parents] + 1, self.root[parents], self.cid[parents])
        self.totals["births"] += valid.sum()

    def _leave_corpses(self, dying, bd):
        """死んだ個体の場所に、体重に応じた肉を残す"""
        c = self.cfg
        dead, slots, valid = self._pair_up(dying, ~self.meat_alive, min(c.max_births_per_step, c.max_meat))
        energy = c.meat_per_mass * bd["flesh"][dead]
        self.meat_pos[slots] = torch.where(valid[:, None], self.pos[dead], self.meat_pos[slots])
        self.meat_energy[slots] = torch.where(valid, energy, self.meat_energy[slots])
        self.meat_alive[slots] = self.meat_alive[slots] | valid

    def spawn_at(self, x, y, n=1, radius=0.0, parent_slot=None):
        """(x, y) の周り radius 以内に生き物を n 匹置く（表示画面のマウス操作から呼ぶ）。
        parent_slot を指定するとその個体の子（突然変異した遺伝子・同じ家族）、なければ完全にランダムな個体"""
        slots = torch.nonzero(~self.alive).squeeze(1)[:n]
        n = slots.shape[0]
        if n == 0:
            return 0
        dev = self.device
        minus1 = torch.full((n,), -1, dtype=torch.long, device=dev)
        if parent_slot is not None and bool(self.alive[parent_slot]):
            p = torch.full((n,), parent_slot, dtype=torch.long, device=dev)
            genes = self.mutate(self.genes[p])
            generation, root, parent = self.generation[p] + 1, self.root[p], self.cid[p]
        else:
            genes = self.random_genes(n)
            generation, root, parent = torch.zeros(n, dtype=torch.long, device=dev), minus1, minus1
        r = radius * torch.rand(n, device=dev).sqrt()
        a = torch.rand(n, device=dev) * 2 * math.pi
        pos = torch.remainder(torch.stack([x + r * torch.cos(a), y + r * torch.sin(a)], -1), self.world_size)
        bd = self.body(self.traits(genes))
        self._spawn(slots, torch.ones(n, dtype=torch.bool, device=dev), genes, pos=pos,
                    theta=(torch.rand(n, device=dev) * 2 - 1) * math.pi, energy=0.6 * bd["emax"], hp=bd["hp_max"],
                    generation=generation, root=root, parent=parent)
        self.totals["spawned"] += n
        return n

    # ================================================================ 植物
    def _sample_food_pos(self, n):
        c = self.cfg
        anywhere = torch.rand(n, 2, device=self.device) * self.world_size
        K = self.hotspots.shape[0]
        if K == 0:  # よく生える場所がなければ、どこにでも均等に生える
            return anywhere
        k = torch.randint(0, K, (n,), device=self.device)
        near = self.hotspots[k] + torch.randn(n, 2, device=self.device) * self.hotspot_r[k, None]
        use_hot = torch.rand(n, 1, device=self.device) < c.hotspot_frac
        return torch.remainder(torch.where(use_hot, near, anywhere), self.world_size)

    # ---- 植物を手で操作する（表示画面から呼ぶ）
    def _wrapped_dist(self, pts, x, y):
        d = torch.remainder(pts - torch.tensor([x, y], device=self.device) + self.world_size / 2,
                            self.world_size) - self.world_size / 2
        return d.norm(dim=-1)

    def _place_food(self, pos):
        free = torch.nonzero(~self.food_alive).squeeze(1)[: len(pos)]
        self.food_pos[free] = pos[: len(free)]
        self.food_alive[free] = True
        return len(free)

    def add_food(self, x, y, radius, n):
        """(x, y) の周り radius 以内に植物を n 個撒く（枠が空いている分だけ）"""
        r = radius * torch.rand(n, device=self.device).sqrt()
        a = torch.rand(n, device=self.device) * 2 * math.pi
        pos = torch.stack([x + r * torch.cos(a), y + r * torch.sin(a)], -1)
        return self._place_food(torch.remainder(pos, self.world_size))

    def remove_food(self, x, y, radius):
        """(x, y) の周り radius 以内の植物を消す"""
        self.food_alive &= self._wrapped_dist(self.food_pos, x, y) >= radius

    def food_rain(self, n=300):
        """よく生える場所の分布に従って、植物をまとめて n 個撒く"""
        return self._place_food(self._sample_food_pos(n))

    def clear_food(self):
        self.food_alive[:] = False

    def add_hotspot(self, x, y, radius):
        self.hotspots = torch.cat([self.hotspots, torch.tensor([[x, y]], device=self.device)])
        self.hotspot_r = torch.cat([self.hotspot_r, torch.tensor([radius], device=self.device)])

    def move_hotspot(self, i, x, y):
        self.hotspots[i] = torch.tensor([x, y], device=self.device) % self.world_size

    def remove_hotspot(self, i):
        keep = torch.arange(self.hotspots.shape[0], device=self.device) != i
        self.hotspots, self.hotspot_r = self.hotspots[keep], self.hotspot_r[keep]

    def hotspot_at(self, x, y):
        """(x, y) が中心付近にある「よく生える場所」の番号（なければ None）"""
        if self.hotspots.shape[0] == 0:
            return None
        d = self._wrapped_dist(self.hotspots, x, y)
        i = int(torch.argmin(d))
        return i if float(d[i]) < max(4.0, 0.5 * float(self.hotspot_r[i])) else None

    def meat_radius(self):
        return 0.5 + 0.12 * self.meat_energy.clamp(min=0).sqrt()

    # ================================================================ 1ステップ
    @torch.no_grad()
    def step(self):
        c = self.cfg
        C = c.max_creatures
        tr = self.traits()
        bd = self.body(tr)
        size, mass, emax, head_sz = bd["size"], bd["mass"], bd["emax"], bd["head"]
        s2 = size.square()
        tail_power, jaw, armor, diet = tr[:, T["tail_power"]], tr[:, T["jaw"]], tr[:, T["armor"]], tr[:, T["diet"]]
        plant_eff = (1 - diet) ** c.diet_specialization
        meat_eff = diet ** c.diet_specialization
        fov, rng = torch.deg2rad(tr[:, T["fov"]]), tr[:, T["sense_range"]]
        alive = self.alive
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        head = self.pos + torch.stack([cos_t, sin_t], -1) * bd["nose"][:, None]
        zeros = torch.zeros(C, device=self.device)

        # ---- 植物を食べる ＋ 植物を見る
        plant_sense, eaten, eater = eat_and_sense_fast(
            head, cos_t, sin_t, fov, rng, self.food_pos, self.food_alive, alive, bd["mouth"], self.food_r,
            self.world_size, c.n_sectors)
        gain = zeros.index_add(0, eater, eaten.float()) * c.food_energy * plant_eff
        self.food_alive = self.food_alive & ~eaten
        self.totals["eaten"] += eaten.sum()

        # ---- 肉（死体）をかじる ＋ 肉を見る
        meat_sense, m_eaten, m_eater = eat_and_sense_fast(
            head, cos_t, sin_t, fov, rng, self.meat_pos, self.meat_alive, alive, bd["mouth"], self.meat_radius(),
            self.world_size, c.n_sectors)
        want = c.meat_bite * s2 * (0.5 + jaw) * head_sz
        taken = torch.where(m_eaten, torch.minimum(self.meat_energy, want[m_eater]), 0.0)
        self.meat_energy = self.meat_energy - taken
        gain = gain + zeros.index_add(0, m_eater, taken) * meat_eff

        # ---- 他の個体を見る ＋ 噛みつく（噛むかどうかは前のステップの脳の判断）
        other_near, who = see_others_fast(head, cos_t, sin_t, fov, rng, self.pos, alive, self.world_size, c.n_sectors)
        seen = other_near > 0
        other_size = torch.where(seen, torch.tanh(torch.log(mass[who] / mass[:, None])), 0.0)  # + なら相手の方が大きい
        other_diet = torch.where(seen, diet[who], 0.0)                                         # 相手が肉食か
        bites, victim = find_bites_fast(head, cos_t, sin_t, self.pos, alive, bd["nose"], bd["body_len"] - bd["nose"],
                                        0.75 * bd["width_max"], bd["mouth"], alive & (self.bite_act > 0.5),
                                        self.world_size)
        dmg = torch.where(bites, c.bite_power * jaw * s2 * head_sz * (1 - c.armor_block * armor[victim]), 0.0)
        hurt = zeros.index_add(0, victim, dmg)
        drained = torch.minimum(zeros.index_add(0, victim, dmg * c.blood_frac), self.energy.clamp(min=0))
        steal = torch.where(bites, dmg * c.blood_frac, 0.0)   # 噛んだ側に移るエネルギー
        gain = gain + steal * meat_eff
        self.hp = self.hp - hurt
        self.hurt = hurt
        self.energy = torch.minimum(self.energy - drained + gain, emax)
        self.food_eaten = self.food_eaten + gain / c.food_energy

        u = self.vel[:, 0] * cos_t + self.vel[:, 1] * sin_t
        w = -self.vel[:, 0] * sin_t + self.vel[:, 1] * cos_t
        x = torch.cat([plant_sense, meat_sense, other_near, other_size, other_diet, torch.stack([
            self.tail, self.tail_vel / c.tail_vmax, u / 0.6, w / 0.3, self.omega / 0.1,
            self.energy / emax, self.hp / bd["hp_max"], torch.sin(self.phase), torch.cos(self.phase)], -1)], -1)

        # ---- 脳（個体ごとに違う重みを一括で計算）
        h = torch.tanh(torch.bmm(x[:, None], self._brain("w_in")).squeeze(1)
                       + torch.bmm(self.hidden[:, None], self._brain("w_h")).squeeze(1) + self._brain("b_h"))
        out = torch.bmm(h[:, None], self._brain("w_out")).squeeze(1) + self._brain("b_out")
        self.hidden, self.last_in, self.last_out = h, x, out
        self.bite_act = torch.sigmoid(out[:, BITE_OUT])

        # ---- 尾びれ: 左右に振ると前に進む
        tail_cmd = torch.tanh(out[:, 0] + tr[:, T["cpg_amp"]] * torch.sin(self.phase))
        tv = self.tail_vel * (1 - c.tail_damp) + c.tail_accel * tail_cmd - c.tail_spring * self.tail
        tv = tv.clamp(-c.tail_vmax, c.tail_vmax)
        tail = self.tail + tv
        tv = torch.where(tail.abs() > 1, 0.0, tv)
        tail = tail.clamp(-1, 1)
        delta = tail - self.tail
        mid = self.tail + 0.5 * delta
        thrust = c.thrust_coef * tail_power * size.pow(2.5) * delta.square() * torch.cos(mid * math.radians(c.tail_max_deg))

        # ---- ヒレ: 向いている方向へ水を押し、体は逆向きに押される（付いている位置で回転も生まれる）
        fin_size = tr[:, self.fin_cols["size"]].clamp(min=0)          # [C, 3]
        fin_pos, fin_ang = tr[:, self.fin_cols["pos"]], torch.deg2rad(tr[:, self.fin_cols["angle"]])
        fin_both = torch.cat([fin_size, fin_size], 1)
        act = torch.sigmoid(out[:, 1:BITE_OUT]) * fin_both.gt(0)      # 生えていないヒレは動かない
        a_l, a_r = act[:, :N_FIN_PAIRS], act[:, N_FIN_PAIRS:]
        mount_x = bd["nose"][:, None] - fin_pos * bd["body_len"][:, None]           # 体の中心からの前後位置
        mount_y = body_width_at(fin_pos, bd["width_max"][:, None], tr[:, T["bulk_pos"], None])
        f = c.fin_force * fin_size * s2[:, None]
        cf, sf = torch.cos(fin_ang), torch.sin(fin_ang)
        fx_l, fy_l = -a_l * f * cf, -a_l * f * sf     # 左ヒレ: 左前方(角度)へ水を押す → 体は逆向き
        fx_r, fy_r = -a_r * f * cf, a_r * f * sf      # 右ヒレ: 左右反転
        force_x = (fx_l + fx_r).sum(1)
        force_y = (fy_l + fy_r).sum(1)
        torque = (mount_x * fy_l - mount_y * fx_l + mount_x * fy_r + mount_y * fx_r).sum(1)
        inertia = mass * (bd["body_len"].square() + (2 * bd["width_max"]).square()) / 12

        # ---- 水の中での動き（体の座標系）。細長い体ほど前後の抵抗が小さい
        drag_f = c.drag_forward * (bd["width"] / bd["length"]).sqrt() + c.fin_idle_drag * fin_size.sum(1)
        u = u * (1 - drag_f) + (thrust + force_x) / mass
        w = w * (1 - c.drag_lateral) + force_y / mass
        omega = (self.omega * (1 - c.drag_angular) + c.fin_rot_gain * torque / inertia
                 + c.tail_yaw_coef * delta / size)
        self.vel = torch.stack([u * cos_t - w * sin_t, u * sin_t + w * cos_t], -1)
        self.theta = torch.remainder(self.theta + omega + math.pi, 2 * math.pi) - math.pi
        self.pos = torch.remainder(self.pos + self.vel, self.world_size)
        self.omega, self.tail, self.tail_vel, self.fin_act = omega, tail, tv, act
        self.phase = torch.remainder(self.phase + 2 * math.pi * tr[:, T["cpg_freq"]], 2 * math.pi)

        # ---- エネルギーを使う・体力が回復する
        cost = (c.basal_cost * mass
                + c.move_cost * tail_power * s2 * delta.square()
                + c.fin_use_cost * (act * fin_both).sum(1) * s2
                + c.fin_keep_cost * fin_size.sum(1) * s2
                + c.jaw_keep_cost * jaw * s2
                + c.armor_keep_cost * armor * mass
                + c.bite_cost * jaw * s2 * (self.bite_act > 0.5)
                + c.sensor_cost * (tr[:, T["fov"]] / 360) * (rng / 70))
        self.energy = self.energy - cost
        self.hp = torch.minimum(self.hp + c.hp_regen * bd["hp_max"], bd["hp_max"])
        self.age = self.age + 1

        # ---- 死ぬ（死体は肉になる）
        dying = alive & ((self.energy <= 0) | (self.age > c.max_age) | (self.hp <= 0))
        killed = dying & (self.hp <= 0)
        self.kills = self.kills + (bites & killed[victim]).long()
        self._leave_corpses(dying, bd)
        self.alive = alive & ~dying
        self.totals["deaths"] += dying.sum()
        self.totals["kills"] += killed.sum()

        # ---- 産む
        self._births(tr, bd)

        # ---- 植物が生える・肉が腐る
        empty = ~self.food_alive
        n_new = c.food_regrow * (c.plant_capacity - self.food_alive.sum()).clamp(min=0)   # 上限に近いほどゆっくり
        grow = empty & (torch.rand(len(empty), device=self.device) < n_new / empty.sum().clamp(min=1))
        self.food_pos = torch.where(grow[:, None], self._sample_food_pos(len(grow)), self.food_pos)
        self.food_alive = self.food_alive | grow
        self.meat_energy = self.meat_energy * (1 - c.meat_decay)
        self.meat_alive = self.meat_alive & (self.meat_energy > 1.0)

        self.steps += 1

    # ================================================================ 表示・記録用
    def summary(self):
        a = self.alive
        n = int(a.sum())
        tr = self.traits()[a]
        out = dict(step=self.steps, population=n, food=int(self.food_alive.sum()),
                   meat=int(self.meat_alive.sum()), **{k: int(v) for k, v in self.totals.items()})
        if n:
            carn = tr[:, T["diet"]] >= 0.5
            out.update(
                gen_max=int(self.generation[a].max()), gen_mean=float(self.generation[a].float().mean()),
                lineages=int(torch.unique(self.root[a]).numel()),
                carnivores=int(carn.sum()), herbivores=int((~carn).sum()),
                trait_mean=tr.mean(0).tolist(), trait_std=tr.std(0).tolist() if n > 1 else [0.0] * N_TRAITS,
                fins_mean=float((tr[:, FIN_SIZE] > 0).float().sum(1).mean() * 2),
                hue_hist=torch.histc(torch.remainder(self.genes[a, HUE], 1.0), bins=36, min=0, max=1).tolist(),
            )
        return out

    RENDER_COLS = ["slot", "x", "y", "theta", "tail", "phase", "energy", "hp", "hurt", "bite", "children",
                   "generation", "hue"] + [f"act{i}" for i in range(2 * N_FIN_PAIRS)] + [name for name, _, _ in TRAITS]

    def render_data(self):
        """描画に必要な値をまとめて CPU に転送する（1回の同期で済ませる）。列名は RENDER_COLS"""
        tr = self.traits()
        bd = self.body(tr)
        pack = torch.cat([
            torch.stack([self.arange_c.float(), self.pos[:, 0], self.pos[:, 1], self.theta, self.tail, self.phase,
                         self.energy / bd["emax"], self.hp / bd["hp_max"], self.hurt, self.bite_act,
                         self.children.float(), self.generation.float(),
                         torch.remainder(self.genes[:, HUE], 1.0)], 1),
            self.fin_act, tr,
        ], 1)[self.alive].cpu().numpy()
        food = self.food_pos[self.food_alive].cpu().numpy()
        meat = torch.cat([self.meat_pos, self.meat_radius()[:, None]], 1)[self.meat_alive].cpu().numpy()
        return pack, food, meat

    def creature_info(self, slot):
        if not bool(self.alive[slot]):
            return None
        tr1 = self.traits()[slot:slot + 1]
        bd = self.body(tr1)
        return dict(
            id=int(self.cid[slot]), root=int(self.root[slot]), generation=int(self.generation[slot]),
            age=int(self.age[slot]), energy=float(self.energy[slot]), energy_max=float(bd["emax"][0]),
            hp=float(self.hp[slot]), hp_max=float(bd["hp_max"][0]),
            food_eaten=float(self.food_eaten[slot]), children=int(self.children[slot]), kills=int(self.kills[slot]),
            traits=tr1[0].tolist(), hue=float(torch.remainder(self.genes[slot, HUE], 1.0)),
            inputs=self.last_in[slot].tolist(), outputs=self.last_out[slot].tolist(),
            fin_act=self.fin_act[slot].tolist(), bite=float(self.bite_act[slot]),
        )

    STATE_KEYS = ("alive", "genes", "pos", "vel", "theta", "omega", "tail", "tail_vel", "fin_act", "bite_act",
                  "phase", "hidden", "energy", "hp", "age", "generation", "food_eaten", "children", "kills", "cid",
                  "root", "parent", "next_id", "hotspots", "hotspot_r", "food_pos", "food_alive", "meat_pos",
                  "meat_energy", "meat_alive")

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {k: getattr(self, k).cpu() for k in self.STATE_KEYS}
        data.update(cfg=asdict(self.cfg), steps=self.steps, version=SAVE_VERSION,
                    totals={k: float(v) for k, v in self.totals.items()})
        torch.save(data, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("version") != SAVE_VERSION:
            raise ValueError(f"{path} は古い形式（今の体のつくりより前）の保存ファイルなので読み込めません")
        known = EvoConfig.__dataclass_fields__   # 今はもう無い設定項目（古い保存ファイル）は無視する
        world = cls(EvoConfig(**{k: v for k, v in data["cfg"].items() if k in known}), device)
        for k in cls.STATE_KEYS:
            if k in data:  # 古い保存ファイルに無い項目は初期値のまま
                setattr(world, k, data[k].to(world.device))
        if "hotspot_r" not in data:
            world.hotspot_r = torch.full((world.hotspots.shape[0],), world.cfg.hotspot_sigma, device=world.device)
        world.steps = data["steps"]
        for k, v in data["totals"].items():
            if k in world.totals:
                world.totals[k] = torch.tensor(v, device=world.device)
        return world
