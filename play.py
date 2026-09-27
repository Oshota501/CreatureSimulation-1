"""
学習した魚を pygame で表示する

    python play.py              # checkpoints/fish_ppo.pt を読み込んで表示
    python play.py --watch      # 学習中にチェックポイントが更新されたら自動で読み直す
    python play.py --manual     # 自分で操作（A/D で尻尾、Q/E でヒレ）

キー操作
    SPACE 一時停止   TAB AI/手動 切替   G 確率的/貪欲 切替
    A / D 尻尾を左/右へ   W 自動で尻尾を振る   Q / E 左/右ヒレを開く
    + / - 再生速度   R リセット   V センサー表示   ESC 終了
"""
import argparse
import math
import os
from collections import deque

import pygame
import torch

from fish_env import TAIL_HOLD, TAIL_LEFT, TAIL_RIGHT, FishConfig, FishEnv
from model import ActorCritic, load_checkpoint

BG_TOP, BG_BOTTOM = (10, 38, 70), (4, 14, 32)
PANEL_BG = (14, 18, 28)
TEXT, DIM, ACCENT = (225, 232, 240), (120, 135, 155), (110, 200, 255)
FOOD, FOOD_GLOW = (150, 255, 120), (90, 220, 90)
BODY, BODY_EDGE, BELLY = (255, 138, 52), (150, 60, 20), (255, 196, 120)
FIN_CLOSED, FIN_OPEN = (255, 170, 90), (255, 235, 110)
SENSOR = (120, 220, 255)
WALL_RAY = (255, 90, 90)
PANEL_W = 330

# 背骨に沿った体の太さ（鼻先 → 尾の付け根）
BODY_PROFILE = [0.35, 0.85, 1.05, 1.1, 1.05, 0.95, 0.82, 0.68, 0.54, 0.42, 0.33, 0.27]
BODY_LEN = 6.4


def rot(x, y, a):
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


class Viewer:
    def __init__(self, args):
        self.args = args
        self.ckpt_path = args.checkpoint
        self.ckpt_mtime = None
        self.ckpt_step = None
        self.model, cfg = self._load_model()
        cfg.episode_len = args.episode_len
        self.env = FishEnv(1, cfg, device="cpu")
        self.cfg = cfg
        self.scale = args.scale
        self.field_w = int(cfg.world_w * self.scale)
        self.field_h = int(cfg.world_h * self.scale)

        pygame.init()
        pygame.display.set_caption("Fish RL - PPO")
        self.screen = pygame.display.set_mode((self.field_w + PANEL_W, self.field_h))
        self.overlay = pygame.Surface((self.field_w, self.field_h), pygame.SRCALPHA)
        self.background = self._make_background()
        mono = pygame.font.match_font("menlo,monaco,dejavusansmono,couriernew")
        self.font = pygame.font.Font(mono, 14)
        self.font_big = pygame.font.Font(mono, 20)
        self.clock = pygame.time.Clock()

        self.manual = args.manual
        self.greedy = False
        self.paused = False
        self.show_sensor = True
        self.sim_speed = 1
        self.auto_wag = False
        self.wag_dir = 1
        self.message, self.message_t = "", 0
        self.reset()

    # ------------------------------------------------------------ モデル
    def _load_model(self):
        if os.path.exists(self.ckpt_path):
            model, cfg, data = load_checkpoint(self.ckpt_path, "cpu")
            self.ckpt_mtime = os.path.getmtime(self.ckpt_path)
            self.ckpt_step = data["global_step"]
            return model, cfg
        print(f"[!] {self.ckpt_path} が無いので、未学習（ランダム）のネットワークで表示します")
        cfg = FishConfig()
        model = ActorCritic(FishEnv(1, cfg).obs_dim).eval()
        return model, cfg

    def _maybe_reload(self):
        if not os.path.exists(self.ckpt_path):
            return
        mtime = os.path.getmtime(self.ckpt_path)
        if mtime == self.ckpt_mtime:
            return
        try:
            model, _, data = load_checkpoint(self.ckpt_path, "cpu")
        except Exception:
            return  # 書き込み途中などは次の機会に
        self.model, self.ckpt_mtime, self.ckpt_step = model, mtime, data["global_step"]
        self.flash(f"reloaded: {self.ckpt_step / 1e6:.1f}M steps")

    def flash(self, msg):
        self.message, self.message_t = msg, 120

    # ------------------------------------------------------------ シミュレーション
    def reset(self):
        self.obs = self.env.reset(stagger=False)
        self.trail = deque(maxlen=240)
        self.effects = []
        self.eaten = 0
        self.steps = 0
        self.action = torch.tensor([[TAIL_HOLD, 0, 0]])

    def manual_action(self):
        keys = pygame.key.get_pressed()
        tail = TAIL_HOLD
        if self.auto_wag:  # 簡単なリズム生成器（CPG）
            t = self.env.tail.item()
            if self.wag_dir > 0 and t > 0.55:
                self.wag_dir = -1
            elif self.wag_dir < 0 and t < -0.55:
                self.wag_dir = 1
            tail = TAIL_LEFT if self.wag_dir > 0 else TAIL_RIGHT
        if keys[pygame.K_a]:
            tail = TAIL_LEFT
        elif keys[pygame.K_d]:
            tail = TAIL_RIGHT
        return torch.tensor([[tail, int(keys[pygame.K_q]), int(keys[pygame.K_e])]])

    def sim_step(self):
        if self.manual:
            self.action = self.manual_action()
        else:
            self.action, _, _ = self.model.act(self.obs, deterministic=self.greedy)
        prev_food = self.env.food[0].clone()
        self.obs, _, done, _ = self.env.step(self.action)
        self.steps += 1
        if self.env.last_eaten[0] > 0:
            self.eaten += int(self.env.last_eaten[0])
            moved = (self.env.food[0] - prev_food).norm(dim=-1) > 1e-6
            for p in prev_food[moved]:
                self.effects.append([p.tolist(), 0])
        self.trail.append(self.env.pos[0].tolist())
        if done[0]:
            self.reset()

    # ------------------------------------------------------------ 座標変換
    def to_screen(self, x, y):
        return x * self.scale, (self.cfg.world_h - y) * self.scale

    def _make_background(self):
        surf = pygame.Surface((self.field_w, self.field_h))
        for y in range(self.field_h):
            k = y / self.field_h
            c = [int(BG_TOP[i] * (1 - k) + BG_BOTTOM[i] * k) for i in range(3)]
            pygame.draw.line(surf, c, (0, y), (self.field_w, y))
        step = int(10 * self.scale)
        grid = pygame.Surface((self.field_w, self.field_h), pygame.SRCALPHA)
        for x in range(0, self.field_w, step):
            pygame.draw.line(grid, (255, 255, 255, 10), (x, 0), (x, self.field_h))
        for y in range(0, self.field_h, step):
            pygame.draw.line(grid, (255, 255, 255, 10), (0, y), (self.field_w, y))
        surf.blit(grid, (0, 0))
        return surf

    # ------------------------------------------------------------ 描画
    def draw(self):
        env, c = self.env, self.cfg
        self.screen.blit(self.background, (0, 0))
        ov = self.overlay
        ov.fill((0, 0, 0, 0))

        pos = env.pos[0].tolist()
        theta = env.theta[0].item()
        head = env.head_pos()[0].tolist()
        obs = self.obs[0]
        food_sense = obs[: c.n_sectors].tolist()
        wall_sense = obs[c.n_sectors : c.n_sectors + env.n_rays].tolist()

        # 軌跡
        if len(self.trail) > 2:
            pts = [self.to_screen(*p) for p in self.trail]
            n = len(pts)
            for i in range(1, n):
                a = int(90 * i / n)
                pygame.draw.line(ov, (180, 230, 255, a), pts[i - 1], pts[i], 2)

        # 正面センサー（扇形）と壁センサー（レイ）
        if self.show_sensor:
            hs = self.to_screen(*head)
            fov = math.radians(c.fov_deg)
            width = fov / c.n_sectors
            for i, v in enumerate(food_sense):
                a0 = theta - fov / 2 + i * width
                arc = [self.to_screen(head[0] + c.sensor_range * math.cos(a0 + width * k / 4),
                                      head[1] + c.sensor_range * math.sin(a0 + width * k / 4)) for k in range(5)]
                alpha = 14 + int(150 * v)
                pygame.draw.polygon(ov, (*SENSOR, alpha), [hs] + arc)
                pygame.draw.polygon(ov, (*SENSOR, 40), [hs] + arc, 1)
            for off, v in zip(c.wall_rays_deg, wall_sense):
                a = theta + math.radians(off)
                d = c.wall_range * (1 - v)
                end = self.to_screen(head[0] + d * math.cos(a), head[1] + d * math.sin(a))
                pygame.draw.line(ov, (*WALL_RAY, 40 + int(200 * v)), hs, end, 1 + int(2 * v))
                if v > 0:
                    pygame.draw.circle(ov, (*WALL_RAY, 200), end, 3)

        # 餌
        r = c.food_radius * self.scale
        for fx, fy in env.food[0].tolist():
            p = self.to_screen(fx, fy)
            pygame.draw.circle(ov, (*FOOD_GLOW, 50), p, r * 2.4)
            pygame.draw.circle(ov, (*FOOD_GLOW, 90), p, r * 1.6)
        # 食べたときのエフェクト
        for e in self.effects:
            e[1] += 1
            k = e[1] / 30
            pygame.draw.circle(ov, (255, 255, 180, int(220 * (1 - k))), self.to_screen(*e[0]),
                               r + k * 5 * self.scale, 3)
        self.effects = [e for e in self.effects if e[1] < 30]

        self.screen.blit(ov, (0, 0))
        for fx, fy in env.food[0].tolist():
            pygame.draw.circle(self.screen, FOOD, self.to_screen(fx, fy), r)

        self.draw_fish(pos, theta, env.tail[0].item(), env.fins[0].tolist())
        self.draw_panel(food_sense, wall_sense)
        pygame.display.flip()

    def draw_fish(self, pos, theta, tail, fins):
        c, s = self.cfg, self.scale
        n = len(BODY_PROFILE)
        tail_max = math.radians(c.tail_max_deg)
        nose = (pos[0] + (c.head_offset + 0.6) * math.cos(theta), pos[1] + (c.head_offset + 0.6) * math.sin(theta))
        # 背骨: 後ろへ行くほど尻尾の角度に沿って曲がる
        spine, angles = [nose], []
        ds = BODY_LEN / (n - 1)
        for i in range(n):
            k = i / (n - 1)
            a = theta + math.pi - tail * tail_max * k * k
            angles.append(a)
            if i < n - 1:
                p = spine[-1]
                spine.append((p[0] + ds * math.cos(a), p[1] + ds * math.sin(a)))
        left, right = [], []
        for p, a, w in zip(spine, angles, BODY_PROFILE):
            nx, ny = math.sin(a), -math.cos(a)  # 魚の左側
            left.append(self.to_screen(p[0] + nx * w, p[1] + ny * w))
            right.append(self.to_screen(p[0] - nx * w, p[1] - ny * w))

        # 尾びれ
        end, ae = spine[-1], angles[-1] - tail * tail_max * 0.5
        def pt(dist, da):
            return self.to_screen(end[0] + dist * math.cos(ae + da), end[1] + dist * math.sin(ae + da))
        caudal = [left[-1], pt(2.6, -0.55), pt(1.3, 0.0), pt(2.6, 0.55), right[-1]]
        pygame.draw.polygon(self.screen, FIN_CLOSED, caudal)
        pygame.draw.polygon(self.screen, BODY_EDGE, caudal, 1)

        # 胸ビレ（開くと横に張り出す）
        i = 3
        for side, is_open in ((1, fins[0] > 0.5), (-1, fins[1] > 0.5)):
            p, a, w = spine[i], angles[i], BODY_PROFILE[i]
            nx, ny = math.sin(a) * side, -math.cos(a) * side
            base = (p[0] + nx * w * 0.8, p[1] + ny * w * 0.8)
            spread = math.radians(80 if is_open else 22)
            length = 2.6 if is_open else 1.8
            fa = a - side * spread
            tip = (base[0] + length * math.cos(fa), base[1] + length * math.sin(fa))
            back = (base[0] + 1.3 * math.cos(a), base[1] + 1.3 * math.sin(a))
            front = (base[0] - 0.3 * math.cos(a), base[1] - 0.3 * math.sin(a))
            poly = [self.to_screen(*front), self.to_screen(*tip), self.to_screen(*back)]
            pygame.draw.polygon(self.screen, FIN_OPEN if is_open else FIN_CLOSED, poly)
            pygame.draw.polygon(self.screen, BODY_EDGE, poly, 1)

        body = left + right[::-1]
        pygame.draw.polygon(self.screen, BODY, body)
        # 背中の模様
        mid = [((l[0] + r[0]) / 2, (l[1] + r[1]) / 2) for l, r in zip(left, right)]
        pygame.draw.lines(self.screen, BELLY, False, mid[1:9], max(1, int(s * 0.35)))
        pygame.draw.polygon(self.screen, BODY_EDGE, body, 2)

        # 目
        p, a = spine[1], angles[1]
        for side in (1, -1):
            nx, ny = math.sin(a) * side, -math.cos(a) * side
            e = self.to_screen(p[0] + nx * 0.5, p[1] + ny * 0.5)
            pygame.draw.circle(self.screen, (255, 255, 255), e, s * 0.32)
            pygame.draw.circle(self.screen, (10, 10, 20), e, s * 0.18)
        # 正面センサー
        pygame.draw.circle(self.screen, SENSOR, self.to_screen(*nose), max(2, s * 0.25))

    def draw_panel(self, food_sense, wall_sense):
        x0 = self.field_w
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, self.field_h))
        y = 16
        def text(t, x=x0 + 18, color=TEXT, font=None):
            nonlocal y
            surf = (font or self.font).render(t, True, color)
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 4

        text("FISH RL  /  PPO", font=self.font_big, color=ACCENT)
        if self.ckpt_step is not None:
            text(f"checkpoint: {self.ckpt_step / 1e6:.1f}M steps", color=DIM)
        else:
            text("checkpoint: none (random)", color=DIM)
        mode = "MANUAL" + (" + auto wag" if self.auto_wag else "") if self.manual else \
            ("AI (greedy)" if self.greedy else "AI (sample)")
        text(f"mode: {mode}" + ("   [PAUSED]" if self.paused else ""))
        text(f"speed: x{self.sim_speed}")
        y += 8
        rate = self.eaten / max(self.steps, 1) * 1000
        text(f"food eaten : {self.eaten}", font=self.font_big)
        text(f"per 1000 st: {rate:.1f}")
        text(f"steps      : {self.steps}")
        y += 10

        # 行動
        a = self.action[0].tolist()
        text("ACTION", color=ACCENT)
        labels = {TAIL_LEFT: "<< LEFT", TAIL_HOLD: " relax ", TAIL_RIGHT: "RIGHT >>"}
        text(f"tail : {labels[a[0]]}")
        bx, bw = x0 + 18, PANEL_W - 36
        pygame.draw.rect(self.screen, (40, 48, 64), (bx, y, bw, 10), border_radius=4)
        tail = self.env.tail[0].item()  # +1 = 左
        tx = bx + bw / 2 - tail * bw / 2
        pygame.draw.line(self.screen, DIM, (bx + bw / 2, y - 2), (bx + bw / 2, y + 12))
        pygame.draw.circle(self.screen, BODY, (tx, y + 5), 7)
        y += 22
        for i, name in enumerate(("L fin", "R fin")):
            on = a[1 + i] == 1
            r = pygame.Rect(x0 + 18 + i * 150, y, 140, 24)
            pygame.draw.rect(self.screen, FIN_OPEN if on else (40, 48, 64), r, border_radius=5)
            s = self.font.render(f"{name}: {'OPEN' if on else 'closed'}", True, (20, 20, 20) if on else DIM)
            self.screen.blit(s, (r.x + 10, r.y + 4))
        y += 36

        # センサー（左から「魚の左」→「魚の右」の順）
        text("FRONT SENSOR (food)", color=ACCENT)
        self._bars(food_sense[::-1], SENSOR, y)
        y += 58
        text("  left" + " " * 22 + "right", color=DIM)
        y += 4
        text("WALL SENSOR", color=ACCENT)
        self._bars(wall_sense[::-1], WALL_RAY, y)
        y += 60

        u = (self.env.vel[0, 0] * math.cos(self.env.theta[0]) + self.env.vel[0, 1] * math.sin(self.env.theta[0])).item()
        text(f"forward speed: {u:5.2f}")
        text(f"turn rate    : {math.degrees(self.env.omega[0].item()):+5.1f} deg/step")

        y = self.field_h - 150
        for line in ("SPACE pause   TAB AI/manual",
                     "G greedy/sample   V sensors",
                     "A/D tail   W auto-wag",
                     "Q/E left/right fin",
                     "+/- sim speed   R reset",
                     "ESC quit"):
            text(line, color=DIM)

        if self.message_t > 0:
            self.message_t -= 1
            s = self.font_big.render(self.message, True, ACCENT)
            self.screen.blit(s, (14, 12))

    def _bars(self, values, color, y):
        n = len(values)
        bx, bw, h = self.field_w + 18, PANEL_W - 36, 50
        w = bw / n
        for i, v in enumerate(values):
            r = pygame.Rect(bx + i * w + 1, y, w - 2, h)
            pygame.draw.rect(self.screen, (32, 38, 52), r, border_radius=3)
            fh = int(h * v)
            if fh > 0:
                pygame.draw.rect(self.screen, color, (r.x, y + h - fh, r.w, fh), border_radius=3)

    # ------------------------------------------------------------ メインループ
    def handle_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_TAB:
            self.manual = not self.manual
        elif key == pygame.K_g:
            self.greedy = not self.greedy
        elif key == pygame.K_v:
            self.show_sensor = not self.show_sensor
        elif key == pygame.K_w:
            self.auto_wag = not self.auto_wag
        elif key == pygame.K_r:
            self.reset()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.sim_speed = min(self.sim_speed * 2, 16)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.sim_speed = max(self.sim_speed // 2, 1)
        return True

    def run(self):
        running, reload_timer = True, 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.handle_key(event.key) and running
            if not self.paused:
                for _ in range(self.sim_speed):
                    self.sim_step()
            reload_timer += 1
            if self.args.watch and reload_timer % 30 == 0:
                self._maybe_reload()
            self.draw()
            self.clock.tick(self.args.fps)
        pygame.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/fish_ppo.pt")
    p.add_argument("--watch", action="store_true", help="チェックポイントが更新されたら読み直す")
    p.add_argument("--manual", action="store_true", help="手動操作モードで起動")
    p.add_argument("--fps", type=int, default=45)
    p.add_argument("--scale", type=float, default=7.5, help="1ワールド単位あたりのピクセル数")
    p.add_argument("--episode-len", type=int, default=10**9)
    Viewer(p.parse_args()).run()


if __name__ == "__main__":
    main()
