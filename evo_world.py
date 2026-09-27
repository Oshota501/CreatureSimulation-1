"""
生命体の進化シミュレーション（GPU 上で全個体・全餌をまとめて計算）

- 各個体は遺伝子（体の形・生き方・脳の重み）を持つ
- 生きているだけ・泳ぐだけでエネルギーが減り、餌を食べると増える
- エネルギーが十分たまったら子を産む。子は親の遺伝子を少しだけ突然変異させて受け継ぐ
- エネルギーが尽きるか寿命が来たら死ぬ
- 個体数が下限を割ったら、これまでで一番餌を食べた個体（殿堂入り）の変異コピーを補充する

座標系は fish_env.py と同じ（x 右、y 上、θ が増えると左に曲がる）。
世界の端はつながっている（右端から出ると左端から入る）。
"""
import math
from dataclasses import asdict, dataclass

import torch

# 体と生き方の遺伝子: (名前, 最小, 最大)。遺伝子の生の値は実数で、sigmoid で範囲に写す
TRAITS = [
    ("size", 0.6, 1.8),           # 体の大きさ: 大きいほど速く口も大きいが、燃費が悪い
    ("tail_power", 0.4, 2.0),     # 尻尾の力: 推進力が上がるが、泳ぐエネルギーも増える
    ("fin_size", 0.3, 2.0),       # 胸ビレ: よく曲がるが、開くと抵抗とエネルギーが増える
    ("fov", 40.0, 320.0),         # センサーの視野角（度）
    ("sense_range", 10.0, 70.0),  # センサーの距離（広く遠いほど維持コストが高い）
    ("cpg_freq", 0.02, 0.2),      # 生まれつきの遊泳リズムの速さ（1ステップあたりの周期数）
    ("cpg_amp", 0.0, 3.0),        # 遊泳リズムが尻尾を動かす強さ
    ("repro_at", 0.4, 0.95),      # エネルギーが最大の何割たまったら子を産むか
    ("child_share", 0.15, 0.6),   # 子に渡すエネルギー（最大エネルギーに対する割合）
    ("mutation", 0.005, 0.12),    # 子に伝わるときの突然変異の強さ（これ自体も進化する）
]
N_TRAITS = len(TRAITS)
T = {name: i for i, (name, _, _) in enumerate(TRAITS)}
HUE = N_TRAITS        # 体色。生存には関係しない目印で、系統ごとに少しずつずれていく
N_MORPH = N_TRAITS + 1
N_OUT = 3             # 脳の出力: 尻尾, 左ヒレ, 右ヒレ


def pair_sense(head, cos_t, sin_t, fov, rng, targets, target_ok, world_size, n_sectors: int, exclude_self: bool):
    """全個体 × 全対象の距離を測り、視野を n_sectors 個の扇形に分けて扇形ごとに一番近いものの近さ (0..1) を返す。
    世界の端はつながっている。角度は + が左。"""
    rel = targets[None, :, :] - head[:, None, :]
    rel = torch.remainder(rel + world_size / 2, world_size) - world_size / 2
    fwd = rel[..., 0] * cos_t[:, None] + rel[..., 1] * sin_t[:, None]
    lat = -rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
    ok = target_ok[None, :]
    if exclude_self:
        idx = torch.arange(head.shape[0], device=head.device)
        ok = ok & (idx[:, None] != idx[None, :])
    dist = torch.where(ok, torch.sqrt(fwd * fwd + lat * lat), torch.inf)
    ang = torch.atan2(lat, fwd)
    half = (fov / 2)[:, None]
    vis = (ang.abs() < half) & (dist < rng[:, None])
    sec = ((ang + half) / fov[:, None] * n_sectors).floor().clamp(0, n_sectors - 1)
    close = torch.where(vis, 1 - dist / rng[:, None], 0.0)
    sense = torch.stack([torch.where(sec == s, close, 0.0).amax(1) for s in range(n_sectors)], 1)
    return sense, dist


def eat_and_sense(head, cos_t, sin_t, fov, rng, food_pos, food_alive, alive, reach, world_size, n_sectors: int):
    """口が触れている餌を決め（1つの餌は一番近い1匹だけが食べる）、同時に餌センサーの値を返す"""
    sense, dist = pair_sense(head, cos_t, sin_t, fov, rng, food_pos, food_alive, world_size, n_sectors, False)
    cand = torch.where((dist < reach[:, None]) & alive[:, None], dist, torch.inf)
    best, eater = cand.min(0)
    return sense, torch.isfinite(best), eater


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
pair_sense_fast = _compiled(pair_sense)


@dataclass
class EvoConfig:
    world_w: float = 400.0
    world_h: float = 300.0
    max_creatures: int = 2048      # 同時に存在できる最大数（GPU 上の枠の数）
    init_creatures: int = 500
    min_creatures: int = 40        # これを下回ったら殿堂入り個体から補充
    hall_size: int = 64
    max_births_per_step: int = 64
    # 餌
    max_food: int = 1600           # 最初は満杯（エデンの園）
    food_energy: float = 25.0
    food_radius: float = 0.8
    food_regrow: float = 0.0015    # 空いている餌の枠1つにつき、1ステップで餌が生える確率
    n_hotspots: int = 4            # 餌がよく生える場所
    hotspot_sigma: float = 30.0
    hotspot_frac: float = 0.7
    # センサーと脳
    n_sectors: int = 7
    hidden: int = 16
    # 泳ぎの物理（fish_env.py と同じモデル）
    tail_accel: float = 0.12
    tail_damp: float = 0.10
    tail_spring: float = 0.02
    tail_vmax: float = 0.45
    tail_max_deg: float = 50.0
    thrust_coef: float = 0.30
    tail_yaw_coef: float = 0.03
    fin_turn_base: float = 0.006
    fin_turn_speed: float = 0.035
    fin_drag: float = 0.04
    drag_forward: float = 0.05
    drag_lateral: float = 0.35
    drag_angular: float = 0.25
    # エネルギー
    energy_per_size2: float = 100.0  # 最大エネルギー = これ × 大きさ²
    basal_cost: float = 0.02         # 生きているだけでかかる（× 大きさ²）
    move_cost: float = 0.5           # 尻尾を振るとかかる（× 尻尾の力 × 大きさ² × 振り²）
    fin_cost: float = 0.004
    sensor_cost: float = 0.03
    birth_overhead: float = 0.1      # 出産時に余分にかかるエネルギーの割合
    min_repro_age: int = 150
    max_age: int = 6000


class World:
    def __init__(self, cfg: EvoConfig | None = None, device="cpu"):
        self.cfg = cfg or EvoConfig()
        self.device = torch.device(device)
        c, dev = self.cfg, self.device
        C, M, S, H = c.max_creatures, c.max_food, c.n_sectors, c.hidden

        self.n_in = 2 * S + 8
        layout = [("w_in", (self.n_in, H)), ("w_h", (H, H)), ("b_h", (H,)), ("w_out", (H, N_OUT)), ("b_out", (N_OUT,))]
        self.brain_layout, off = {}, N_MORPH
        for name, shape in layout:
            n = math.prod(shape)
            self.brain_layout[name] = (off, off + n, shape)
            off += n
        self.n_genes = off
        self.lo = torch.tensor([t[1] for t in TRAITS], device=dev)
        self.hi = torch.tensor([t[2] for t in TRAITS], device=dev)
        self.world_size = torch.tensor([c.world_w, c.world_h], device=dev)
        self.arange_c = torch.arange(C, device=dev)

        zf = lambda *s: torch.zeros(*s, device=dev)
        zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        # 個体
        self.alive = torch.zeros(C, dtype=torch.bool, device=dev)
        self.genes = zf(C, self.n_genes)
        self.pos, self.vel = zf(C, 2), zf(C, 2)
        self.theta, self.omega = zf(C), zf(C)
        self.tail, self.tail_vel, self.fins = zf(C), zf(C), zf(C, 2)
        self.phase, self.hidden = zf(C), zf(C, H)
        self.energy, self.age = zf(C), zl(C)
        self.generation, self.food_eaten, self.children = zl(C), zf(C), zl(C)
        self.cid, self.root, self.parent = zl(C), zl(C), zl(C)
        self.last_in, self.last_out = zf(C, self.n_in), zf(C, N_OUT)
        self.next_id = torch.zeros((), dtype=torch.long, device=dev)
        # 餌
        self.hotspots = torch.rand(c.n_hotspots, 2, device=dev) * self.world_size
        self.food_pos = self._sample_food_pos(M)
        self.food_alive = torch.ones(M, dtype=torch.bool, device=dev)
        # 殿堂入り（死んだ個体のうち、一生で食べた餌が多かったもの）
        self.hall_genes = zf(c.hall_size, self.n_genes)
        self.hall_score = torch.full((c.hall_size,), -1.0, device=dev)
        self.hall_root, self.hall_gen = zl(c.hall_size), zl(c.hall_size)

        self.steps = 0
        self.totals = {k: torch.zeros((), device=dev) for k in ("births", "deaths", "eaten", "injected")}

        n = c.init_creatures
        slots = self.arange_c[:n]
        genes = self.random_genes(n)
        size = self.traits(genes)[:, T["size"]]
        self._spawn(slots, torch.ones(n, dtype=torch.bool, device=dev), genes,
                    pos=torch.rand(n, 2, device=dev) * self.world_size,
                    theta=(torch.rand(n, device=dev) * 2 - 1) * math.pi,
                    energy=0.6 * c.energy_per_size2 * size.square(),
                    generation=zl(n), root=torch.full((n,), -1, device=dev, dtype=torch.long),
                    parent=torch.full((n,), -1, device=dev, dtype=torch.long))

    # ================================================================ 遺伝子
    def traits(self, genes=None):
        g = self.genes if genes is None else genes
        return self.lo + (self.hi - self.lo) * torch.sigmoid(g[:, :N_TRAITS])

    def random_genes(self, n):
        g = torch.zeros(n, self.n_genes, device=self.device)
        g[:, :N_TRAITS] = torch.randn(n, N_TRAITS, device=self.device)
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
    def _spawn(self, slots, valid, genes, pos, theta, energy, generation, root, parent):
        """slots（重複なし）のうち valid な枠に新しい個体を書き込む"""
        ids = self.next_id + torch.cumsum(valid.long(), 0) - 1
        self.next_id = self.next_id + valid.sum()
        n = slots.shape[0]
        zf = lambda *s: torch.zeros(*s, device=self.device)
        values = dict(
            genes=genes, pos=pos, vel=zf(n, 2), theta=theta, omega=zf(n), tail=zf(n), tail_vel=zf(n),
            fins=zf(n, 2), phase=torch.rand(n, device=self.device) * 2 * math.pi, hidden=zf(n, self.cfg.hidden),
            energy=energy, age=torch.zeros_like(generation), generation=generation, food_eaten=zf(n),
            children=torch.zeros_like(generation), cid=ids, root=torch.where(root < 0, ids, root), parent=parent,
            alive=torch.ones(n, dtype=torch.bool, device=self.device),
        )
        for name, v in values.items():
            cur = getattr(self, name)
            old = cur[slots]
            m = valid.view(-1, *([1] * (old.dim() - 1)))
            cur[slots] = torch.where(m, v.to(cur.dtype), old)

    def _births(self, tr, emax):
        c = self.cfg
        B = c.max_births_per_step
        ready = self.alive & (self.energy > tr[:, T["repro_at"]] * emax) & (self.age > c.min_repro_age)
        free = ~self.alive
        # 「産める親」と「空いている枠」をそれぞれ先頭に並べ、k 番目どうしを組にする（GPU 上で完結）
        parents = torch.argsort((~ready).long() * len(ready) + self.arange_c)[:B]
        slots = torch.argsort((~free).long() * len(free) + self.arange_c)[:B]
        k = self.arange_c[:B]
        valid = (k < ready.sum()) & (k < free.sum())

        genes = self.mutate(self.genes[parents])
        child_size = self.traits(genes)[:, T["size"]]
        e_parent = self.energy[parents]
        e_child = torch.minimum(tr[parents, T["child_share"]] * emax[parents], 0.8 * e_parent / (1 + c.birth_overhead))
        e_child = torch.minimum(e_child, c.energy_per_size2 * child_size.square())
        self.energy[parents] = e_parent - torch.where(valid, e_child * (1 + c.birth_overhead), 0.0)
        self.children[parents] = self.children[parents] + valid.long()

        th = self.theta[parents]
        back = torch.stack([torch.cos(th), torch.sin(th)], -1) * (4 * tr[parents, T["size"]])[:, None]
        pos = torch.remainder(self.pos[parents] - back + torch.randn(B, 2, device=self.device), self.world_size)
        self._spawn(slots, valid, genes, pos, th + torch.randn(B, device=self.device) * 0.5, e_child,
                    self.generation[parents] + 1, self.root[parents], self.cid[parents])
        self.totals["births"] += valid.sum()

    def _update_hall(self, dying):
        """死んだ個体のうち成績の良いものを殿堂入りリストに残す"""
        Hs = self.cfg.hall_size
        score = torch.where(dying, self.food_eaten + 0.5 * self.children, -1.0)
        top_s, top_i = torch.cat([self.hall_score, score]).topk(Hs)
        from_hall = top_i < Hs
        hi, ci = top_i.clamp(max=Hs - 1), (top_i - Hs).clamp(min=0)
        self.hall_genes = torch.where(from_hall[:, None], self.hall_genes[hi], self.genes[ci])
        self.hall_root = torch.where(from_hall, self.hall_root[hi], self.root[ci])
        self.hall_gen = torch.where(from_hall, self.hall_gen[hi], self.generation[ci])
        self.hall_score = top_s

    def _maintain_population(self):
        """個体数が下限を割ったら補充する（序盤の補助輪）"""
        c = self.cfg
        n_alive = int(self.alive.sum())
        if n_alive >= c.min_creatures:
            return
        n = c.min_creatures - n_alive + c.min_creatures // 2
        slots = torch.nonzero(~self.alive).squeeze(1)[:n]
        n = slots.shape[0]
        good = self.hall_score > 0
        use_hall = good.any() & (torch.rand(n, device=self.device) < 0.8)
        pick = torch.multinomial(good.float() + 1e-6, n, replacement=True)
        genes = torch.where(use_hall[:, None], self.mutate(self.hall_genes[pick]), self.random_genes(n))
        size = self.traits(genes)[:, T["size"]]
        minus1 = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self._spawn(slots, torch.ones(n, dtype=torch.bool, device=self.device), genes,
                    pos=torch.rand(n, 2, device=self.device) * self.world_size,
                    theta=(torch.rand(n, device=self.device) * 2 - 1) * math.pi,
                    energy=0.6 * c.energy_per_size2 * size.square(),
                    generation=torch.where(use_hall, self.hall_gen[pick], 0),
                    root=torch.where(use_hall, self.hall_root[pick], minus1), parent=minus1)
        self.totals["injected"] += n

    # ================================================================ 餌
    def _sample_food_pos(self, n):
        c = self.cfg
        k = torch.randint(0, c.n_hotspots, (n,), device=self.device)
        near = self.hotspots[k] + torch.randn(n, 2, device=self.device) * c.hotspot_sigma
        anywhere = torch.rand(n, 2, device=self.device) * self.world_size
        use_hot = torch.rand(n, 1, device=self.device) < c.hotspot_frac
        return torch.remainder(torch.where(use_hot, near, anywhere), self.world_size)

    # ================================================================ 1ステップ
    @torch.no_grad()
    def step(self):
        c = self.cfg
        tr = self.traits()
        size, tail_power, fin_size = tr[:, T["size"]], tr[:, T["tail_power"]], tr[:, T["fin_size"]]
        fov, rng = torch.deg2rad(tr[:, T["fov"]]), tr[:, T["sense_range"]]
        emax = c.energy_per_size2 * size.square()
        alive = self.alive
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        head = self.pos + torch.stack([cos_t, sin_t], -1) * (2 * size)[:, None]

        # ---- 食べる（口が餌に触れたら）＋ 正面センサーで餌と他の個体を見る
        food_sense, eaten, eater = eat_and_sense_fast(
            head, cos_t, sin_t, fov, rng, self.food_pos, self.food_alive, alive,
            1.2 * size + c.food_radius, self.world_size, c.n_sectors)
        got = torch.zeros(len(alive), device=self.device).index_add_(0, eater, eaten.float())
        self.energy = torch.minimum(self.energy + got * c.food_energy, emax)
        self.food_eaten = self.food_eaten + got
        self.food_alive = self.food_alive & ~eaten
        self.totals["eaten"] += eaten.sum()
        other_sense, _ = pair_sense_fast(head, cos_t, sin_t, fov, rng, self.pos, alive, self.world_size,
                                         c.n_sectors, True)

        u = self.vel[:, 0] * cos_t + self.vel[:, 1] * sin_t
        w = -self.vel[:, 0] * sin_t + self.vel[:, 1] * cos_t
        x = torch.cat([food_sense, other_sense, torch.stack([
            self.tail, self.tail_vel / c.tail_vmax, u / 0.6, w / 0.3, self.omega / 0.1,
            self.energy / emax, torch.sin(self.phase), torch.cos(self.phase)], -1)], -1)

        # ---- 脳（個体ごとに違う重みを一括で計算）
        h = torch.tanh(torch.bmm(x[:, None], self._brain("w_in")).squeeze(1)
                       + torch.bmm(self.hidden[:, None], self._brain("w_h")).squeeze(1) + self._brain("b_h"))
        out = torch.bmm(h[:, None], self._brain("w_out")).squeeze(1) + self._brain("b_out")
        self.hidden, self.last_in, self.last_out = h, x, out

        # ---- 体を動かす
        tail_cmd = torch.tanh(out[:, 0] + tr[:, T["cpg_amp"]] * torch.sin(self.phase))
        fins = torch.sigmoid(out[:, 1:3])
        tv = self.tail_vel * (1 - c.tail_damp) + c.tail_accel * tail_cmd - c.tail_spring * self.tail
        tv = tv.clamp(-c.tail_vmax, c.tail_vmax)
        tail = self.tail + tv
        tv = torch.where(tail.abs() > 1, 0.0, tv)
        tail = tail.clamp(-1, 1)
        delta = tail - self.tail
        mid = self.tail + 0.5 * delta
        thrust = c.thrust_coef * tail_power * size.sqrt() * delta.square() * torch.cos(mid * math.radians(c.tail_max_deg))

        u = u * (1 - c.drag_forward - c.fin_drag * fin_size * fins.sum(1)) + thrust
        w = w * (1 - c.drag_lateral)
        turn = (fins[:, 0] - fins[:, 1]) * fin_size / size * (c.fin_turn_base + c.fin_turn_speed * u.abs())
        omega = self.omega * (1 - c.drag_angular) + turn + c.tail_yaw_coef * delta / size
        self.vel = torch.stack([u * cos_t - w * sin_t, u * sin_t + w * cos_t], -1)
        self.theta = torch.remainder(self.theta + omega + math.pi, 2 * math.pi) - math.pi
        self.pos = torch.remainder(self.pos + self.vel, self.world_size)
        self.omega, self.tail, self.tail_vel, self.fins = omega, tail, tv, fins
        self.phase = torch.remainder(self.phase + 2 * math.pi * tr[:, T["cpg_freq"]], 2 * math.pi)

        # ---- エネルギーを使う
        cost = (c.basal_cost * size.square()
                + c.move_cost * tail_power * size.square() * delta.square()
                + c.fin_cost * fin_size * fins.sum(1)
                + c.sensor_cost * (tr[:, T["fov"]] / 360) * (rng / 70))
        self.energy = self.energy - cost
        self.age = self.age + 1

        # ---- 死ぬ
        dying = alive & ((self.energy <= 0) | (self.age > c.max_age))
        self._update_hall(dying)
        self.alive = alive & ~dying
        self.totals["deaths"] += dying.sum()

        # ---- 産む
        self._births(tr, emax)

        # ---- 餌が生える
        grow = ~self.food_alive & (torch.rand(len(self.food_alive), device=self.device) < c.food_regrow)
        self.food_pos = torch.where(grow[:, None], self._sample_food_pos(len(grow)), self.food_pos)
        self.food_alive = self.food_alive | grow

        self.steps += 1
        if self.steps % 25 == 0:
            self._maintain_population()

    # ================================================================ 表示・記録用
    def summary(self):
        a = self.alive
        n = int(a.sum())
        tr = self.traits()[a]
        out = dict(step=self.steps, population=n, food=int(self.food_alive.sum()),
                   **{k: int(v) for k, v in self.totals.items()})
        if n:
            out.update(
                gen_max=int(self.generation[a].max()), gen_mean=float(self.generation[a].float().mean()),
                lineages=int(torch.unique(self.root[a]).numel()),
                trait_mean=tr.mean(0).tolist(), trait_std=tr.std(0).tolist() if n > 1 else [0.0] * N_TRAITS,
                hue_hist=torch.histc(torch.remainder(self.genes[a, HUE], 1.0), bins=36, min=0, max=1).tolist(),
            )
        return out

    def render_data(self):
        """描画に必要な値をまとめて CPU に転送する（1回の同期で済ませる）"""
        tr = self.traits()
        pack = torch.cat([
            self.arange_c[:, None].float(), self.pos, self.theta[:, None], self.tail[:, None], self.fins,
            tr[:, [T["size"], T["fin_size"], T["fov"], T["sense_range"]]],
            torch.remainder(self.genes[:, HUE:HUE + 1], 1.0),
            (self.energy / (self.cfg.energy_per_size2 * tr[:, T["size"]].square()))[:, None],
        ], 1)[self.alive].cpu().numpy()
        food = self.food_pos[self.food_alive].cpu().numpy()
        return pack, food

    def creature_info(self, slot):
        if not bool(self.alive[slot]):
            return None
        tr = self.traits()[slot].tolist()
        return dict(
            id=int(self.cid[slot]), root=int(self.root[slot]), generation=int(self.generation[slot]),
            age=int(self.age[slot]), energy=float(self.energy[slot]),
            energy_max=self.cfg.energy_per_size2 * tr[T["size"]] ** 2,
            food_eaten=int(self.food_eaten[slot]), children=int(self.children[slot]),
            traits=tr, hue=float(torch.remainder(self.genes[slot, HUE], 1.0)),
            inputs=self.last_in[slot].tolist(), outputs=self.last_out[slot].tolist(),
        )

    STATE_KEYS = ("alive", "genes", "pos", "vel", "theta", "omega", "tail", "tail_vel", "fins", "phase", "hidden",
                  "energy", "age", "generation", "food_eaten", "children", "cid", "root", "parent", "next_id",
                  "hotspots", "food_pos", "food_alive", "hall_genes", "hall_score", "hall_root", "hall_gen")

    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {k: getattr(self, k).cpu() for k in self.STATE_KEYS}
        data.update(cfg=asdict(self.cfg), steps=self.steps, totals={k: float(v) for k, v in self.totals.items()})
        torch.save(data, path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        world = cls(EvoConfig(**data["cfg"]), device)
        for k in cls.STATE_KEYS:
            setattr(world, k, data[k].to(world.device))
        world.steps = data["steps"]
        world.totals = {k: torch.tensor(v, device=world.device) for k, v in data["totals"].items()}
        return world
