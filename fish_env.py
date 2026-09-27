"""
魚のシミュレーション環境（PyTorch でベクトル化し、GPU 上で大量の水槽を同時に動かす）

座標系: x が右、y が上（数学座標）。heading θ=0 で右向き。
θ が増える = 反時計回り = 魚から見て「左」に曲がる。

行動 (MultiDiscrete [3, 2, 2])
    action[:, 0] 尻尾   0=左へ振る / 1=力を抜く / 2=右へ振る
    action[:, 1] 左ヒレ 0=閉じる / 1=開く
    action[:, 2] 右ヒレ 0=閉じる / 1=開く

観測
    [正面センサー: 視野を n_sectors 個の扇形に分け、各扇形で一番近い餌の近さ (0..1)]
    [壁センサー: 頭から数本のレイを飛ばし、壁の近さ (0..1)]
    [自分の状態: 尻尾の角度, 尻尾の速度, 前進速度, 横滑り速度, 旋回速度]
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

TAIL_LEFT, TAIL_HOLD, TAIL_RIGHT = 0, 1, 2
ACTION_DIMS = (3, 2, 2)
N_PROPRIO = 5


@dataclass
class FishConfig:
    # フィールド
    world_w: float = 100.0
    world_h: float = 100.0
    wall_margin: float = 2.0      # 体の中心が壁に近づける最小距離
    n_food: int = 6
    food_radius: float = 1.0
    food_margin: float = 4.0      # 餌は壁からこの距離以上内側に出現
    episode_len: int = 1000
    # 体
    head_offset: float = 2.0      # 体の中心 → 頭（口と正面センサー）までの距離
    mouth_radius: float = 1.2
    # 正面センサー（餌の方向を検知）
    fov_deg: float = 120.0
    n_sectors: int = 11
    sensor_range: float = 45.0
    # 壁センサー（レイ）
    wall_rays_deg: tuple = (-60.0, -30.0, 0.0, 30.0, 60.0)
    wall_range: float = 20.0
    # 尻尾: 慣性つき。振りの速さ^2 に比例して推進力が出る
    tail_accel: float = 0.12
    tail_damp: float = 0.10
    tail_spring: float = 0.02     # 力を抜くとまっすぐに戻ろうとする
    tail_vmax: float = 0.45
    tail_max_deg: float = 50.0    # 尻尾角度 ±1 に対応する実角度
    thrust_coef: float = 0.30
    tail_yaw_coef: float = 0.03   # 尻尾を振った反動で頭が振られる
    # ヒレ: 開いた側に抵抗が生まれ、その側へ曲がる
    fin_turn_base: float = 0.006  # 止まっていても少しは回れる
    fin_turn_speed: float = 0.035 # 速いほどよく曲がる
    fin_drag: float = 0.04        # 開くとブレーキにもなる
    # 水の抵抗
    drag_forward: float = 0.05
    drag_lateral: float = 0.35
    drag_angular: float = 0.25
    # 報酬
    food_reward: float = 1.0
    approach_coef: float = 0.0    # >0 で「餌に近づいたら少し報酬」の補助報酬を追加
    wall_penalty: float = 0.0


class FishEnv:
    """num_envs 個の独立した水槽（それぞれ魚1匹 + 餌 n_food 個）をまとめて扱う"""

    def __init__(self, num_envs: int, cfg: FishConfig | None = None, device="cpu"):
        self.cfg = cfg or FishConfig()
        self.n = num_envs
        self.device = torch.device(device)
        c = self.cfg
        self.n_rays = len(c.wall_rays_deg)
        self.obs_dim = c.n_sectors + self.n_rays + N_PROPRIO
        self.ray_offsets = torch.tensor([math.radians(a) for a in c.wall_rays_deg], device=self.device)
        self.fov = math.radians(c.fov_deg)
        self.tail_max_rad = math.radians(c.tail_max_deg)

        z = lambda *s: torch.zeros(*s, device=self.device)
        self.pos, self.vel = z(self.n, 2), z(self.n, 2)
        self.theta, self.omega = z(self.n), z(self.n)
        self.tail, self.tail_vel = z(self.n), z(self.n)
        self.fins = z(self.n, 2)
        self.food = z(self.n, c.n_food, 2)
        self.near_dist = z(self.n)
        self.ep_return = z(self.n)
        self.t = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.last_eaten = z(self.n)
        self.reset_stats()

    # ------------------------------------------------------------------ 初期化
    def _rand_xy(self, *shape, margin):
        c = self.cfg
        u = torch.rand(*shape, 2, device=self.device)
        return torch.stack([margin + u[..., 0] * (c.world_w - 2 * margin),
                            margin + u[..., 1] * (c.world_h - 2 * margin)], -1)

    def _fresh_state(self):
        c = self.cfg
        n = self.n
        return dict(
            pos=self._rand_xy(n, margin=c.wall_margin + 8.0),
            vel=torch.zeros(n, 2, device=self.device),
            theta=(torch.rand(n, device=self.device) * 2 - 1) * math.pi,
            omega=torch.zeros(n, device=self.device),
            tail=(torch.rand(n, device=self.device) * 2 - 1) * 0.3,
            tail_vel=torch.zeros(n, device=self.device),
            fins=torch.zeros(n, 2, device=self.device),
            food=self._rand_xy(n, c.n_food, margin=c.food_margin),
        )

    def reset(self, stagger=True):
        """全水槽をリセット。stagger=True だとエピソードの終わるタイミングをばらけさせる"""
        for k, v in self._fresh_state().items():
            setattr(self, k, v)
        if stagger:
            self.t = torch.randint(0, self.cfg.episode_len, (self.n,), device=self.device)
        else:
            self.t.zero_()
        self.ep_return.zero_()
        self.near_dist = self._nearest_food_dist()
        return self._obs()

    def _reset_where(self, mask):
        for k, v in self._fresh_state().items():
            cur = getattr(self, k)
            m = mask.view(-1, *([1] * (cur.dim() - 1)))
            setattr(self, k, torch.where(m, v, cur))
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return), self.ep_return)
        self.near_dist = torch.where(mask, self._nearest_food_dist(), self.near_dist)

    def reset_stats(self):
        z = lambda: torch.zeros((), device=self.device)
        self.stats = dict(eaten=z(), steps=z(), speed=z(), wall=z(), ep_return=z(), ep_count=z())

    # ------------------------------------------------------------------ 補助
    def head_pos(self, pos=None, theta=None):
        pos = self.pos if pos is None else pos
        theta = self.theta if theta is None else theta
        return pos + torch.stack([torch.cos(theta), torch.sin(theta)], -1) * self.cfg.head_offset

    def _nearest_food_dist(self):
        return (self.food - self.head_pos()[:, None, :]).norm(dim=-1).amin(1)

    # ------------------------------------------------------------------ 1ステップ
    def step(self, action):
        c = self.cfg
        a_tail = action[:, 0]
        fins = action[:, 1:3].to(torch.float32)
        cmd = torch.where(a_tail == TAIL_LEFT, 1.0, torch.where(a_tail == TAIL_RIGHT, -1.0, 0.0))

        # --- 尻尾（左右に振るほど前へ進む） ---
        tv = self.tail_vel * (1 - c.tail_damp) + c.tail_accel * cmd - c.tail_spring * self.tail
        tv = tv.clamp(-c.tail_vmax, c.tail_vmax)
        tail = self.tail + tv
        hit_limit = tail.abs() > 1.0
        tail = tail.clamp(-1.0, 1.0)
        tv = torch.where(hit_limit, torch.zeros_like(tv), tv)
        delta = tail - self.tail
        mid = self.tail + 0.5 * delta
        # 真ん中付近を速く通過するほど推進力が大きい（小刻みに震わせても進まない）
        thrust = c.thrust_coef * delta.square() * torch.cos(mid * self.tail_max_rad)

        # --- 体の座標系で速度を更新（前後は抵抗小、横は抵抗大） ---
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        u = self.vel[:, 0] * cos_t + self.vel[:, 1] * sin_t
        w = -self.vel[:, 0] * sin_t + self.vel[:, 1] * cos_t
        u = u * (1 - c.drag_forward - c.fin_drag * fins.sum(1)) + thrust
        w = w * (1 - c.drag_lateral)

        # --- 旋回: 左ヒレを開く → 左に曲がる / 右ヒレ → 右 ---
        turn = (fins[:, 0] - fins[:, 1]) * (c.fin_turn_base + c.fin_turn_speed * u.abs())
        omega = self.omega * (1 - c.drag_angular) + turn + c.tail_yaw_coef * delta

        vel = torch.stack([u * cos_t - w * sin_t, u * sin_t + w * cos_t], -1)
        theta = torch.remainder(self.theta + omega + math.pi, 2 * math.pi) - math.pi
        pos = self.pos + vel

        # --- 壁 ---
        lo = c.wall_margin
        hit_x = (pos[:, 0] < lo) | (pos[:, 0] > c.world_w - lo)
        hit_y = (pos[:, 1] < lo) | (pos[:, 1] > c.world_h - lo)
        pos = torch.stack([pos[:, 0].clamp(lo, c.world_w - lo), pos[:, 1].clamp(lo, c.world_h - lo)], -1)
        vel = torch.stack([torch.where(hit_x, 0.0, vel[:, 0]), torch.where(hit_y, 0.0, vel[:, 1])], -1)
        wall_hit = (hit_x | hit_y).float()

        self.pos, self.vel, self.theta, self.omega = pos, vel, theta, omega
        self.tail, self.tail_vel, self.fins = tail, tv, fins

        # --- 餌を食べたら報酬、餌は別の場所に再出現 ---
        head = self.head_pos()
        dist = (self.food - head[:, None, :]).norm(dim=-1)
        eaten = dist < (c.mouth_radius + c.food_radius)
        n_eaten = eaten.sum(1).float()
        self.food = torch.where(eaten[..., None], self._rand_xy(self.n, c.n_food, margin=c.food_margin), self.food)
        reward = c.food_reward * n_eaten - c.wall_penalty * wall_hit
        if c.approach_coef > 0:
            near = self._nearest_food_dist()
            reward = reward + c.approach_coef * (self.near_dist - near) * (n_eaten == 0)
            self.near_dist = near

        self.last_eaten = n_eaten
        self.ep_return = self.ep_return + reward
        self.t = self.t + 1
        done = self.t >= c.episode_len

        s = self.stats
        s["eaten"] += n_eaten.sum()
        s["steps"] += self.n
        s["speed"] += u.abs().sum()
        s["wall"] += wall_hit.sum()
        s["ep_return"] += (self.ep_return * done).sum()
        s["ep_count"] += done.sum()

        # 時間切れの水槽はリセット。terminal_obs は価値のブートストラップ用
        terminal_obs = self._obs()
        self._reset_where(done)
        obs = torch.where(done[:, None], self._obs(), terminal_obs)
        return obs, reward, done, terminal_obs

    # ------------------------------------------------------------------ センサー
    def _obs(self):
        c = self.cfg
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        head = self.head_pos()

        # 正面センサー: 視野角内の餌を扇形ごとに集計（近いほど 1 に近い）
        rel = self.food - head[:, None, :]
        fwd = rel[..., 0] * cos_t[:, None] + rel[..., 1] * sin_t[:, None]
        lat = -rel[..., 0] * sin_t[:, None] + rel[..., 1] * cos_t[:, None]
        dist = torch.sqrt(fwd.square() + lat.square())
        ang = torch.atan2(lat, fwd)  # + が左
        half = self.fov / 2
        visible = (ang.abs() < half) & (dist < c.sensor_range)
        idx = ((ang + half) / self.fov * c.n_sectors).floor().clamp(0, c.n_sectors - 1).long()
        closeness = (1 - dist / c.sensor_range).clamp(min=0) * visible
        food_sense = (F.one_hot(idx, c.n_sectors) * closeness[..., None]).amax(1)

        # 壁センサー: レイと壁の交点までの距離
        ra = self.theta[:, None] + self.ray_offsets[None, :]
        dx, dy = torch.cos(ra), torch.sin(ra)
        hx, hy = head[:, 0:1], head[:, 1:2]
        big = torch.full_like(dx, 1e6)
        tx = torch.where(dx > 1e-6, (c.world_w - hx) / dx, torch.where(dx < -1e-6, -hx / dx, big))
        ty = torch.where(dy > 1e-6, (c.world_h - hy) / dy, torch.where(dy < -1e-6, -hy / dy, big))
        wall_d = torch.minimum(tx, ty).clamp(min=0)
        wall_sense = 1 - (wall_d / c.wall_range).clamp(max=1)

        u = self.vel[:, 0] * cos_t + self.vel[:, 1] * sin_t
        w = -self.vel[:, 0] * sin_t + self.vel[:, 1] * cos_t
        proprio = torch.stack([self.tail, self.tail_vel / c.tail_vmax, u / 0.6, w / 0.3, self.omega / 0.1], -1)
        return torch.cat([food_sense, wall_sense, proprio], -1)
