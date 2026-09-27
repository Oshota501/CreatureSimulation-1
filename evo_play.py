"""
生命体の進化シミュレーションを pygame で表示する

    python evo_play.py                            # 新しい世界で開始
    python evo_play.py --load saves/world.pt      # 保存した世界の続きから
    python evo_play.py --headless --steps 50000   # 画面なしで高速に進めて保存

操作
    SPACE 一時停止    + / - 再生速度    マウスホイール ズーム    右ドラッグ / 矢印キー 移動
    左クリック 個体を選択    F 選択した個体を追いかける    C カメラを戻す
    R 餌の雨（餌を満杯にする）    H 餌の多い場所の表示    S 保存    ESC 終了
"""
import argparse
import math
import os
import time

import numpy as np
import pygame
import torch

from evo_world import TRAITS, EvoConfig, World

VIEW_W, VIEW_H, PANEL_W = 1000, 750, 340
SPEEDS = [1, 2, 4, 8, 16, 32, 0]  # 0 = 最大速度
BG_TOP, BG_BOTTOM = (12, 40, 72), (4, 14, 32)
PANEL_BG, TEXT, DIM, ACCENT = (14, 18, 28), (225, 232, 240), (120, 135, 155), (110, 200, 255)
FOOD = (150, 255, 120)
SENSOR, OTHER = (120, 220, 255), (255, 170, 90)
TRAIT_LABELS = {
    "size": "body size", "tail_power": "tail power", "fin_size": "fin size", "fov": "sensor FOV",
    "sense_range": "sensor range", "cpg_freq": "swim rhythm", "cpg_amp": "rhythm power",
    "repro_at": "breed at", "child_share": "child share", "mutation": "mutation",
}
BODY_WIDTH = np.array([0.45, 1.05, 0.95, 0.6, 0.3])


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hsv_to_rgb(h, s, v):
    """numpy でまとめて HSV → RGB (0..255)"""
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], -1) * 255).clip(0, 255).astype(int)


def fish_shapes(p, tail_max):
    """全個体の体・尾びれ・胸ビレの頂点を計算（カメラ中心からの相対ワールド座標）"""
    dx, dy, th, tail = p["dx"], p["dy"], p["theta"], p["tail"]
    size, fin_size = p["size"], p["fin_size"]
    n, K = len(dx), len(BODY_WIDTH)
    ks = np.arange(K) / (K - 1)
    ang = th[:, None] + np.pi - tail[:, None] * tail_max * ks[None] ** 2
    px, py = np.empty((n, K)), np.empty((n, K))
    px[:, 0] = dx + 2.6 * size * np.cos(th)
    py[:, 0] = dy + 2.6 * size * np.sin(th)
    seg = 1.5 * size
    for k in range(1, K):
        px[:, k] = px[:, k - 1] + seg * np.cos(ang[:, k - 1])
        py[:, k] = py[:, k - 1] + seg * np.sin(ang[:, k - 1])
    wid = BODY_WIDTH[None] * size[:, None]
    nx, ny = np.sin(ang), -np.cos(ang)  # 魚の左側
    lx, ly, rx, ry = px + nx * wid, py + ny * wid, px - nx * wid, py - ny * wid
    body = np.concatenate([np.stack([lx, ly], -1), np.stack([rx, ry], -1)[:, ::-1]], 1)

    ae = ang[:, -1] - tail * tail_max * 0.5
    ex, ey = px[:, -1], py[:, -1]
    def at(dist, da):
        return np.stack([ex + dist * size * np.cos(ae + da), ey + dist * size * np.sin(ae + da)], -1)
    caudal = np.stack([np.stack([lx[:, -1], ly[:, -1]], -1), at(2.2, -0.55), at(1.1, 0.0), at(2.2, 0.55),
                       np.stack([rx[:, -1], ry[:, -1]], -1)], 1)

    fins = []
    a1, w1 = ang[:, 1], wid[:, 1]
    for side, f in ((1, p["fin_l"]), (-1, p["fin_r"])):
        bx, by = px[:, 1] + side * nx[:, 1] * w1 * 0.8, py[:, 1] + side * ny[:, 1] * w1 * 0.8
        spread = math.radians(20) + math.radians(60) * f
        length = (1.0 + 0.8 * f) * fin_size * size
        fa = a1 - side * spread
        tip = np.stack([bx + length * np.cos(fa), by + length * np.sin(fa)], -1)
        back = np.stack([bx + size * np.cos(a1), by + size * np.sin(a1)], -1)
        front = np.stack([bx - 0.3 * size * np.cos(a1), by - 0.3 * size * np.sin(a1)], -1)
        fins.append(np.stack([front, tip, back], 1))
    return body, caudal, fins[0], fins[1], np.stack([px[:, 0], py[:, 0]], -1)


class Viewer:
    def __init__(self, world: World, args):
        self.world, self.args = world, args
        self.cfg = world.cfg
        pygame.init()
        pygame.display.set_caption("Evolution")
        self.screen = pygame.display.set_mode((VIEW_W + PANEL_W, VIEW_H))
        self.overlay = pygame.Surface((VIEW_W, VIEW_H), pygame.SRCALPHA)
        self.bg = pygame.Surface((VIEW_W, VIEW_H))
        for y in range(VIEW_H):
            k = y / VIEW_H
            pygame.draw.line(self.bg, [int(BG_TOP[i] * (1 - k) + BG_BOTTOM[i] * k) for i in range(3)], (0, y), (VIEW_W, y))
        mono = pygame.font.match_font("menlo,monaco,dejavusansmono,couriernew")
        self.font = pygame.font.Font(mono, 13)
        self.font_big = pygame.font.Font(mono, 18)
        self.clock = pygame.time.Clock()

        self.fit_zoom = min(VIEW_W / self.cfg.world_w, VIEW_H / self.cfg.world_h)
        self.reset_camera()
        self.speed_i = 2
        self.paused = False
        self.show_hotspots = True
        self.selected = None
        self.follow = False
        self.dragging = None
        self.summary = world.summary()
        self.history = []
        self.frame = 0
        self.message, self.message_t = "", 0
        self.sps, self._sps_t, self._sps_steps = 0.0, time.time(), world.steps
        self.food_sprites = {}

    # ------------------------------------------------------------ カメラ
    def reset_camera(self):
        self.cam = np.array([self.cfg.world_w / 2, self.cfg.world_h / 2])
        self.zoom = self.fit_zoom

    def wrap_rel(self, x, y):
        """カメラ中心からの相対位置（世界の端はつながっている）"""
        W, H = self.cfg.world_w, self.cfg.world_h
        return (x - self.cam[0] + W / 2) % W - W / 2, (y - self.cam[1] + H / 2) % H - H / 2

    def to_screen(self, dx, dy):
        return dx * self.zoom + VIEW_W / 2, VIEW_H / 2 - dy * self.zoom

    def flash(self, msg):
        self.message, self.message_t = msg, 150

    # ------------------------------------------------------------ 入力
    def handle(self, event):
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_ESCAPE:
                return False
            if k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed_i = max(self.speed_i - 1, 0)
            elif k == pygame.K_f and self.selected is not None:
                self.follow = not self.follow
            elif k == pygame.K_c:
                self.reset_camera()
                self.follow = False
            elif k == pygame.K_h:
                self.show_hotspots = not self.show_hotspots
            elif k == pygame.K_r:
                self.world.food_alive[:] = True
                self.flash("food rain!")
            elif k == pygame.K_s:
                self.world.save(self.args.save)
                self.flash(f"saved: {self.args.save}")
        elif event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if mx < VIEW_W:
                before = np.array([(mx - VIEW_W / 2) / self.zoom, (VIEW_H / 2 - my) / self.zoom])
                self.zoom = float(np.clip(self.zoom * (1.15 ** event.y), self.fit_zoom, self.fit_zoom * 10))
                after = np.array([(mx - VIEW_W / 2) / self.zoom, (VIEW_H / 2 - my) / self.zoom])
                if not self.follow:
                    self.cam += before - after
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (2, 3):
                self.dragging = event.pos
            elif event.button == 1 and event.pos[0] < VIEW_W:
                self.select_at(*event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (2, 3):
            self.dragging = None
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            dx, dy = event.pos[0] - self.dragging[0], event.pos[1] - self.dragging[1]
            self.cam += np.array([-dx / self.zoom, dy / self.zoom])
            self.dragging = event.pos
            self.follow = False
        return True

    def select_at(self, mx, my):
        p = self.pack
        if not len(p["slot"]):
            return
        sx, sy = self.to_screen(p["dx"], p["dy"])
        d = np.hypot(sx - mx, sy - my)
        i = int(np.argmin(d))
        if d[i] < max(12, 4 * p["size"][i] * self.zoom):
            self.selected = int(p["slot"][i])
        else:
            self.selected, self.follow = None, False

    # ------------------------------------------------------------ シミュレーション
    def advance(self):
        if self.paused:
            return
        n = SPEEDS[self.speed_i]
        if n:
            for _ in range(n):
                self.world.step()
        else:  # 最大速度: 1フレームあたり約 50ms 分進める
            t = time.time()
            while time.time() - t < 0.05:
                self.world.step()

    def refresh(self):
        data, food = self.world.render_data()
        data, food = data.astype(np.float64), food.astype(np.float64)  # pygame は float32 を受け付けない
        keys = ["slot", "x", "y", "theta", "tail", "fin_l", "fin_r", "size", "fin_size", "fov", "range", "hue", "energy"]
        p = {k: data[:, i] for i, k in enumerate(keys)}
        p["slot"] = p["slot"].astype(int)
        if self.follow and self.selected is not None:
            hit = np.nonzero(p["slot"] == self.selected)[0]
            if len(hit):
                self.cam = np.array([p["x"][hit[0]], p["y"][hit[0]]])
        p["dx"], p["dy"] = self.wrap_rel(p["x"], p["y"])
        self.pack, self.food = p, food
        if self.frame % 12 == 0:
            self.summary = self.world.summary()
            s = self.summary
            if not self.history or s["step"] > self.history[-1][0]:
                self.history.append((s["step"], s["population"], s["food"]))
                self.history = self.history[-600:]
        now = time.time()
        if now - self._sps_t > 1.0:
            self.sps = (self.world.steps - self._sps_steps) / (now - self._sps_t)
            self._sps_t, self._sps_steps = now, self.world.steps
        self.info = self.world.creature_info(self.selected) if self.selected is not None else None
        if self.selected is not None and self.info is None:
            self.flash("the selected creature died")
            self.selected, self.follow = None, False

    # ------------------------------------------------------------ 描画
    def food_sprite(self):
        r = max(1.5, self.cfg.food_radius * self.zoom)
        key = round(r * 2) / 2
        if key not in self.food_sprites:
            R = int(math.ceil(key * 2.5))
            s = pygame.Surface((2 * R + 1, 2 * R + 1), pygame.SRCALPHA)
            pygame.draw.circle(s, (90, 220, 90, 45), (R, R), R)
            pygame.draw.circle(s, FOOD, (R, R), key)
            self.food_sprites[key] = (s, R)
        return self.food_sprites[key]

    def draw_world(self):
        self.screen.blit(self.bg, (0, 0))
        ov = self.overlay
        ov.fill((0, 0, 0, 0))
        W, H = self.cfg.world_w, self.cfg.world_h

        if self.show_hotspots:
            for hx, hy in self.world.hotspots.cpu().double().numpy():
                dx, dy = self.wrap_rel(hx, hy)
                for ox in (-W, 0, W):
                    for oy in (-H, 0, H):
                        c = self.to_screen(dx + ox, dy + oy)
                        for k, a in ((1.6, 10), (1.0, 12), (0.5, 14)):
                            pygame.draw.circle(ov, (60, 200, 110, a), c, self.cfg.hotspot_sigma * k * self.zoom)

        # 境界線（つながっている世界の継ぎ目）
        x0, y0 = self.to_screen(*self.wrap_rel(0.0, 0.0))
        pygame.draw.line(ov, (255, 255, 255, 25), (x0, 0), (x0, VIEW_H))
        pygame.draw.line(ov, (255, 255, 255, 25), (0, y0), (VIEW_W, y0))

        p = self.pack
        sel = None
        if self.selected is not None:
            hit = np.nonzero(p["slot"] == self.selected)[0]
            sel = int(hit[0]) if len(hit) else None
        if sel is not None and self.info is not None:
            self.draw_sensor(sel)
        self.screen.blit(ov, (0, 0))

        # 餌
        if len(self.food):
            sprite, R = self.food_sprite()
            fx, fy = self.wrap_rel(self.food[:, 0], self.food[:, 1])
            sx, sy = self.to_screen(fx, fy)
            vis = (sx > -5) & (sx < VIEW_W + 5) & (sy > -5) & (sy < VIEW_H + 5)
            self.screen.blits([(sprite, (x - R, y - R)) for x, y in zip(sx[vis].tolist(), sy[vis].tolist())], False)

        # 生き物
        if not len(p["slot"]):
            return
        sx, sy = self.to_screen(p["dx"], p["dy"])
        margin = 12 * self.zoom
        vis = np.nonzero((sx > -margin) & (sx < VIEW_W + margin) & (sy > -margin) & (sy < VIEW_H + margin))[0]
        if not len(vis):
            return
        q = {k: v[vis] for k, v in p.items()}
        body, caudal, fin_l, fin_r, nose = fish_shapes(q, math.radians(self.cfg.tail_max_deg))
        conv = lambda a: np.stack([a[..., 0] * self.zoom + VIEW_W / 2, VIEW_H / 2 - a[..., 1] * self.zoom], -1).tolist()
        body, caudal, fin_l, fin_r, nose = map(conv, (body, caudal, fin_l, fin_r, nose))
        v = 0.62 + 0.38 * np.clip(q["energy"], 0, 1)  # お腹が空いているほど暗い
        col_body = hsv_to_rgb(q["hue"], np.full_like(v, 0.62), v).tolist()
        col_fin = hsv_to_rgb(q["hue"], np.full_like(v, 0.35), np.minimum(v + 0.1, 1)).tolist()
        col_edge = hsv_to_rgb(q["hue"], np.full_like(v, 0.7), v * 0.45).tolist()
        detail = self.zoom * np.median(q["size"]) > 3.0
        draw = pygame.draw.polygon
        for i in range(len(vis)):
            draw(self.screen, col_fin[i], caudal[i])
            if detail:
                draw(self.screen, col_fin[i], fin_l[i])
                draw(self.screen, col_fin[i], fin_r[i])
            draw(self.screen, col_body[i], body[i])
            if detail:
                draw(self.screen, col_edge[i], body[i], 1)
        if sel is not None:
            j = np.nonzero(vis == sel)[0]
            if len(j):
                c = self.to_screen(p["dx"][sel], p["dy"][sel])
                pygame.draw.circle(self.screen, (255, 255, 255), c, 7 * p["size"][sel] * self.zoom / 2 + 6, 2)

    def draw_sensor(self, i):
        p, info, c = self.pack, self.info, self.cfg
        S = c.n_sectors
        th, size = p["theta"][i], p["size"][i]
        hx, hy = p["dx"][i] + 2 * size * math.cos(th), p["dy"][i] + 2 * size * math.sin(th)
        fov, rng = math.radians(p["fov"][i]), p["range"][i]
        food, other = info["inputs"][:S], info["inputs"][S:2 * S]
        hs = self.to_screen(hx, hy)
        wdt = fov / S
        for s in range(S):
            a0 = th - fov / 2 + s * wdt
            arc = [self.to_screen(hx + rng * math.cos(a0 + wdt * k / 4), hy + rng * math.sin(a0 + wdt * k / 4))
                   for k in range(5)]
            pygame.draw.polygon(self.overlay, (*SENSOR, 12 + int(140 * food[s])), [hs] + arc)
            if other[s] > 0:
                d = rng * (1 - other[s])
                am = a0 + wdt / 2
                pygame.draw.circle(self.overlay, (*OTHER, 200), self.to_screen(hx + d * math.cos(am), hy + d * math.sin(am)), 4)
            pygame.draw.polygon(self.overlay, (*SENSOR, 45), [hs] + arc, 1)

    # ------------------------------------------------------------ パネル
    def draw_panel(self):
        x0 = VIEW_W
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, PANEL_W, VIEW_H))
        s = self.summary
        self.y = 12
        spd = SPEEDS[self.speed_i]
        self.text("EVOLUTION", font=self.font_big, color=ACCENT)
        self.text(f"step {s['step']:,}   {'MAX' if spd == 0 else f'x{spd}'}   {self.sps:,.0f} st/s"
                  + ("   [PAUSED]" if self.paused else ""), color=DIM)
        self.y += 4
        self.text(f"population {s['population']:5d}   food {s['food']:5d}", font=self.font_big)
        self.text(f"generation max {s.get('gen_max', 0)}  mean {s.get('gen_mean', 0):.1f}")
        self.text(f"lineages   {s.get('lineages', 0)}   (founder families alive)")
        self.text(f"births {s['births']:,}  deaths {s['deaths']:,}  rescued {s['injected']}", color=DIM)
        self.y += 6
        self.draw_graph()
        self.y += 8
        self.draw_hue_strip()
        self.y += 10
        if self.info is not None:
            self.draw_creature()
        else:
            self.draw_traits()
        yy = VIEW_H - 84
        for line in ("SPACE pause   +/- speed   wheel zoom",
                     "right-drag / arrows: move camera",
                     "click select   F follow   C camera",
                     "R food rain   H hotspots   S save"):
            self.screen.blit(self.font.render(line, True, DIM), (x0 + 14, yy))
            yy += 18
        if self.message_t > 0:
            self.message_t -= 1
            self.screen.blit(self.font_big.render(self.message, True, ACCENT), (14, 12))

    def text(self, t, color=TEXT, font=None, x=14):
        surf = (font or self.font).render(t, True, color)
        self.screen.blit(surf, (VIEW_W + x, self.y))
        self.y += surf.get_height() + 3

    def draw_graph(self):
        x, w, h = VIEW_W + 14, PANEL_W - 28, 90
        r = pygame.Rect(x, self.y, w, h)
        pygame.draw.rect(self.screen, (24, 30, 44), r, border_radius=4)
        if len(self.history) > 1:
            hist = np.array(self.history, dtype=float)
            top = max(hist[:, 1].max(), hist[:, 2].max(), 1)
            t0, t1 = hist[0, 0], max(hist[-1, 0], hist[0, 0] + 1)
            for col, color in ((2, (90, 200, 90)), (1, (255, 200, 90))):
                pts = [(x + (t - t0) / (t1 - t0) * w, self.y + h - 4 - v / top * (h - 10)) for t, v in zip(hist[:, 0], hist[:, col])]
                pygame.draw.lines(self.screen, color, False, pts, 2)
            self.screen.blit(self.font.render(f"{int(top)}", True, DIM), (x + 4, self.y + 2))
        self.screen.blit(self.font.render("population", True, (255, 200, 90)), (x + w - 170, self.y + 2))
        self.screen.blit(self.font.render("food", True, (90, 200, 90)), (x + w - 44, self.y + 2))
        self.y += h + 4

    def draw_hue_strip(self):
        hist = self.summary.get("hue_hist")
        x, w, h = VIEW_W + 14, PANEL_W - 28, 30
        self.text("body colors (lineage markers)", color=DIM)
        if not hist:
            self.y += h
            return
        n, top = len(hist), max(max(hist), 1)
        cols = hsv_to_rgb((np.arange(n) + 0.5) / n, np.full(n, 0.62), np.full(n, 0.95)).tolist()
        bw = w / n
        for i, v in enumerate(hist):
            bh = int(h * (v / top) ** 0.5)
            if bh:
                pygame.draw.rect(self.screen, cols[i], (x + i * bw, self.y + h - bh, bw - 1, bh))
        pygame.draw.line(self.screen, (60, 70, 90), (x, self.y + h), (x + w, self.y + h))
        self.y += h + 6

    def draw_traits(self):
        s = self.summary
        self.text("POPULATION AVERAGE  (bar = mean +- std)", color=ACCENT)
        if "trait_mean" not in s:
            return
        x, w = VIEW_W + 14, PANEL_W - 28
        for i, (name, lo, hi) in enumerate(TRAITS):
            m, sd = s["trait_mean"][i], s["trait_std"][i]
            self.screen.blit(self.font.render(TRAIT_LABELS[name], True, TEXT), (x, self.y))
            self.screen.blit(self.font.render(f"{m:7.2f}", True, DIM), (x + w - 58, self.y))
            bx, bw = x + 112, w - 176
            by = self.y + 5
            pygame.draw.rect(self.screen, (36, 44, 60), (bx, by, bw, 6), border_radius=3)
            a = (max(m - sd, lo) - lo) / (hi - lo)
            b = (min(m + sd, hi) - lo) / (hi - lo)
            pygame.draw.rect(self.screen, (70, 110, 150), (bx + a * bw, by, max(2, (b - a) * bw), 6), border_radius=3)
            pygame.draw.circle(self.screen, ACCENT, (bx + (m - lo) / (hi - lo) * bw, by + 3), 4)
            self.y += 19

    def draw_creature(self):
        info = self.info
        x, w = VIEW_W + 14, PANEL_W - 28
        col = hsv_to_rgb(np.array([info["hue"]]), np.array([0.62]), np.array([0.95]))[0].tolist()
        pygame.draw.rect(self.screen, col, (x, self.y + 2, 12, 12), border_radius=3)
        self.text(f"   CREATURE #{info['id']}" + ("  (following)" if self.follow else ""), color=ACCENT)
        self.text(f"generation {info['generation']}   age {info['age']}   family #{info['root']}")
        self.text(f"food eaten {info['food_eaten']}   children {info['children']}")
        e = info["energy"] / info["energy_max"]
        self.text(f"energy {info['energy']:.0f} / {info['energy_max']:.0f}")
        pygame.draw.rect(self.screen, (36, 44, 60), (x, self.y, w, 8), border_radius=3)
        pygame.draw.rect(self.screen, (120, 230, 120), (x, self.y, max(0, e) * w, 8), border_radius=3)
        self.y += 14
        tr = info["traits"]
        for i in range(0, len(TRAITS), 2):
            parts = []
            for j in (i, i + 1):
                if j < len(TRAITS):
                    parts.append(f"{TRAIT_LABELS[TRAITS[j][0]][:12]:<12} {tr[j]:6.2f}")
            self.text("  ".join(parts), color=DIM)
        self.y += 4
        S = self.cfg.n_sectors
        self.text("SENSOR  food (blue) / others (orange)", color=ACCENT)
        self.bars(info["inputs"][:S][::-1], SENSOR, 26)
        self.bars(info["inputs"][S:2 * S][::-1], OTHER, 26)
        right = self.font.render("right", True, DIM)
        self.screen.blit(right, (VIEW_W + PANEL_W - 14 - right.get_width(), self.y))
        self.text("left", color=DIM)
        out = info["outputs"]
        fins = [1 / (1 + math.exp(-o)) for o in out[1:3]]
        self.text(f"brain: tail {out[0]:+.2f}  fin L {fins[0]:.2f} R {fins[1]:.2f}")

    def bars(self, values, color, h):
        x, w = VIEW_W + 14, PANEL_W - 28
        n = len(values)
        bw = w / n
        for i, v in enumerate(values):
            pygame.draw.rect(self.screen, (32, 38, 52), (x + i * bw + 1, self.y, bw - 2, h), border_radius=3)
            fh = int(h * v)
            if fh:
                pygame.draw.rect(self.screen, color, (x + i * bw + 1, self.y + h - fh, bw - 2, fh), border_radius=3)
        self.y += h + 4

    # ------------------------------------------------------------ メインループ
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle(event) and running
            keys = pygame.key.get_pressed()
            pan = 8 / self.zoom
            move = np.array([keys[pygame.K_RIGHT] - keys[pygame.K_LEFT], keys[pygame.K_UP] - keys[pygame.K_DOWN]])
            if move.any():
                self.cam += move * pan
                self.follow = False
            self.advance()
            self.refresh()
            self.draw_world()
            self.draw_panel()
            pygame.display.flip()
            self.frame += 1
            self.clock.tick(60)
        pygame.quit()


def run_headless(world, args):
    t0 = time.time()
    for i in range(args.steps):
        world.step()
        if (i + 1) % 2000 == 0 or i + 1 == args.steps:
            s = world.summary()
            tm = s.get("trait_mean", [0.0] * len(TRAITS))
            print(f"step {s['step']:7d}  {s['step'] / (time.time() - t0):5.0f} st/s  pop {s['population']:4d}  "
                  f"food {s['food']:4d}  gen {s.get('gen_max', 0):4d}  lineages {s.get('lineages', 0):3d}  |  "
                  + "  ".join(f"{TRAIT_LABELS[n]}={v:.2f}" for (n, _, _), v in zip(TRAITS, tm)), flush=True)
    world.save(args.save)
    print(f"saved: {args.save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto / cuda / mps / cpu")
    p.add_argument("--load", help="保存した世界ファイル")
    p.add_argument("--save", default="saves/world.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="画面なしで --steps だけ進めて保存")
    p.add_argument("--steps", type=int, default=50000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    world = World.load(args.load, device) if args.load else World(EvoConfig(), device)
    print(f"device={device}  population={int(world.alive.sum())}  step={world.steps}")
    if args.headless:
        run_headless(world, args)
    else:
        Viewer(world, args).run()


if __name__ == "__main__":
    main()
