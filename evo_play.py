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
import time

import numpy as np
import pygame
import torch

from evo_world import N_FIN_PAIRS, TRAITS, EvoConfig, World

VIEW_W, VIEW_H, PANEL_W = 1000, 800, 340
SPEEDS = [1, 2, 4, 8, 16, 32, 0]  # 0 = 最大速度
BG_TOP, BG_BOTTOM = (12, 40, 72), (4, 14, 32)
PANEL_BG, BOX_BG, TEXT, DIM, ACCENT = (14, 18, 28), (22, 30, 46), (225, 232, 240), (120, 135, 155), (110, 200, 255)
FOOD = (150, 255, 120)
SENSOR, OTHER = (120, 220, 255), (255, 170, 90)
TRAIT_LABELS = {
    "size": "size", "body_len": "length", "body_wid": "girth", "bulk_pos": "belly pos",
    "tail_power": "tail fin", "tail_fork": "tail fork", "fov": "eye FOV", "sense_range": "eye range",
    "cpg_freq": "rhythm", "cpg_amp": "rhythm pow", "repro_at": "breed at", "child_share": "child shr",
    "mutation": "mutation", "stripes": "stripes", "pattern_hue": "pattern",
    "head_size": "head", "jaw": "jaw", "armor": "armor", "diet": "diet",
}
MEAT, MEAT_EDGE = (150, 40, 40), (90, 20, 25)
HERB_COL, CARN_COL, PLANT_COL = (255, 200, 90), (255, 90, 90), (90, 200, 90)
BODY_TRAITS = [name for name, _, _ in TRAITS if not name.startswith("fin")]
TRAIT_RANGE = {name: (lo, hi) for name, lo, hi in TRAITS}
TRAIT_INDEX = {name: i for i, (name, _, _) in enumerate(TRAITS)}
K_SPINE = 8


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
    h = np.remainder(h, 1.0)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], -1) * 255).clip(0, 255).astype(int)


def fin_words(pos, angle):
    where = "face" if pos < 0.25 else ("mid" if pos < 0.65 else "rear")
    way = "fwd" if angle < 60 else ("side" if angle < 120 else "back")
    return where, way


def creature_geometry(q, tail_max):
    """全個体の体・尾びれ・ヒレ・縞・目の頂点を計算する（q["dx"], q["dy"] を中心にしたワールド座標）"""
    n = len(q["dx"])
    th, tail, size = q["theta"], q["tail"], q["size"]
    body_len = 6.0 * size * q["body_len"]
    width_max = 1.05 * size * q["body_wid"]
    p_exp = np.log(0.5) / np.log(q["bulk_pos"])
    ts = np.linspace(0, 1, K_SPINE)
    ang = th[:, None] + np.pi - tail[:, None] * tail_max * ts[None] ** 2   # 背骨の各点での「後ろ向き」の角度
    seg = body_len / (K_SPINE - 1)
    px, py = np.empty((n, K_SPINE)), np.empty((n, K_SPINE))
    px[:, 0] = q["dx"] + 0.43 * body_len * np.cos(th)
    py[:, 0] = q["dy"] + 0.43 * body_len * np.sin(th)
    for k in range(1, K_SPINE):
        px[:, k] = px[:, k - 1] + seg * np.cos(ang[:, k - 1])
        py[:, k] = py[:, k - 1] + seg * np.sin(ang[:, k - 1])

    head_extra = 0.55 * (q["head_size"] - 1.0)

    def width(t):
        base = 0.18 + 0.82 * np.sin(np.pi * np.clip(t, 1e-4, 1) ** p_exp[:, None])
        base = base + head_extra[:, None] * np.clip(1 - t / 0.35, 0, 1)   # 頭が大きいほど頭まわりが太い
        return width_max[:, None] * np.maximum(base, 0.12)

    def at(t):
        """背骨上の位置 t (0=顔, 1=尾の付け根) の座標と「後ろ向き」の角度"""
        f = t * (K_SPINE - 1)
        j = np.clip(np.floor(f).astype(int), 0, K_SPINE - 2)
        f = f - j
        lerp = lambda a: np.take_along_axis(a, j, 1) * (1 - f) + np.take_along_axis(a, j + 1, 1) * f
        return lerp(px), lerp(py), lerp(ang), width(t)

    wid = width(np.broadcast_to(ts, (n, K_SPINE)))
    nx, ny = np.sin(ang), -np.cos(ang)  # 魚の左側
    left = np.stack([px + nx * wid, py + ny * wid], -1)
    right = np.stack([px - nx * wid, py - ny * wid], -1)
    body = np.concatenate([left, right[:, ::-1]], 1)

    # 尾びれ: 大きさは尾の力、切れ込みは tail_fork
    tp = q["tail_power"]
    ae = ang[:, -1] - tail * tail_max * 0.5
    ex, ey = px[:, -1], py[:, -1]
    length_c = (0.9 + 0.9 * tp) * size
    spread = 0.35 + 0.25 * tp
    notch = length_c * (1 - 0.65 * q["tail_fork"])
    def tip(d, da):
        return np.stack([ex + d * np.cos(ae + da), ey + d * np.sin(ae + da)], -1)
    caudal = np.stack([left[:, -1], tip(length_c, -spread), tip(notch, 0 * ae), tip(length_c, spread), right[:, -1]], 1)

    # ヒレ: 付いている位置・向き・大きさは遺伝子どおり。使っているとパタパタ動く
    fins, fin_on, fin_act = [], [], []
    for i in range(N_FIN_PAIRS):
        fs = np.clip(q[f"fin{i}_size"], 0, None)
        fx0, fy0, fa, w0 = (v[:, 0] for v in at(q[f"fin{i}_pos"][:, None]))
        psi = np.radians(q[f"fin{i}_angle"])
        length = 2.0 * fs * size
        for side, k in ((1, i), (-1, N_FIN_PAIRS + i)):
            a = q[f"act{k}"]
            flap = a * 0.45 * np.sin(q["phase"] * 3 + i * 1.3 + (side > 0) * 0.8)
            bx = fx0 + side * np.sin(fa) * w0 * 0.85
            by = fy0 - side * np.cos(fa) * w0 * 0.85
            phi = fa + np.pi + side * (psi + flap)
            dx, dy = np.cos(phi), np.sin(phi)
            rx, ry = -dy * side, dx * side
            wb = 0.28 * length
            pts = [(bx + rx * wb, by + ry * wb),
                   (bx + dx * 0.55 * length + rx * 0.28 * length, by + dy * 0.55 * length + ry * 0.28 * length),
                   (bx + dx * length, by + dy * length),
                   (bx + dx * 0.55 * length - rx * 0.1 * length, by + dy * 0.55 * length - ry * 0.1 * length),
                   (bx - rx * wb, by - ry * wb)]
            fins.append(np.stack([np.stack(p, -1) for p in pts], 1))
            fin_on.append(fs > 0.02)
            fin_act.append(a)

    # 縞: 0〜4 本
    n_str = np.floor(q["stripes"]).astype(int)
    j = np.arange(4)
    tj = 0.25 + (j[None] + 0.5) / np.maximum(n_str[:, None], 1) * 0.5
    sx, sy, sa, sw = at(tj)
    sw = sw * 0.95
    snx, sny = np.sin(sa), -np.cos(sa)
    stripes = np.stack([np.stack([sx + snx * sw, sy + sny * sw], -1), np.stack([sx - snx * sw, sy - sny * sw], -1)], 2)

    # 目: 視野が広いほど横に、遠くまで見えるほど大きく
    exx, eyy, ea, we = (v[:, 0] for v in at(np.full((n, 1), 0.1)))
    lat = (0.25 + 0.55 * (q["fov"] - 40) / 280) * we
    r_eye = size * (0.15 + 0.2 * (q["sense_range"] - 10) / 60)
    eyes = np.stack([np.stack([exx + s * np.sin(ea) * lat, eyy - s * np.cos(ea) * lat], -1) for s in (1, -1)], 1)
    fwd = ea + np.pi
    pupils = eyes + (0.35 * r_eye[:, None] * np.stack([np.cos(fwd), np.sin(fwd)], -1))[:, None, :]

    # 顎（キバ）: 攻撃力が強いほど大きく、噛むときに開く
    jaw = q["jaw"]
    nx0, ny0 = px[:, 0], py[:, 0]
    fwd0 = ang[:, 0] + np.pi
    w_nose = wid[:, 1] * 0.8
    length_j = size * q["head_size"] * (0.3 + 1.1 * jaw / 1.5)
    opening = 0.12 + 0.55 * q["bite"]
    jaws = []
    for s in (1, -1):
        lx, ly = np.sin(ang[:, 0]) * s, -np.cos(ang[:, 0]) * s   # その側の外向き
        base_out = np.stack([nx0 + lx * w_nose - np.cos(fwd0) * 0.3 * size, ny0 + ly * w_nose - np.sin(fwd0) * 0.3 * size], -1)
        base_in = np.stack([nx0 + lx * w_nose * 0.15, ny0 + ly * w_nose * 0.15], -1)
        a = fwd0 + s * opening
        tip = np.stack([nx0 + lx * w_nose * 0.5 + np.cos(a) * length_j, ny0 + ly * w_nose * 0.5 + np.sin(a) * length_j], -1)
        hook = tip - np.stack([lx, ly], -1) * (0.35 * length_j)[:, None] - np.stack([np.cos(a), np.sin(a)], -1) * (0.2 * length_j)[:, None]
        jaws.append(np.stack([base_out, tip, hook, base_in], 1))

    # 鎧（トゲ）: 鎧が強いほど本数が多く長い
    n_spk = np.round(q["armor"] * 6).astype(int)
    kk = np.arange(6)
    tk = 0.22 + (kk[None] + 0.5) / np.maximum(n_spk[:, None], 1) * 0.6
    kx, ky, ka, kw = at(tk)
    spk_len = (0.4 + 0.9 * q["armor"]) * size
    spikes = []
    for s in (1, -1):
        ox, oy = np.sin(ka) * s, -np.cos(ka) * s
        bx, by = kx + ox * kw * 0.9, ky + oy * kw * 0.9
        back = ka   # 後ろ向き
        dx_, dy_ = ox * 0.8 + np.cos(back) * 0.6, oy * 0.8 + np.sin(back) * 0.6
        tipx, tipy = bx + dx_ * spk_len[:, None], by + dy_ * spk_len[:, None]
        h = 0.22 * spk_len[:, None]
        spikes.append(np.stack([np.stack([bx - np.cos(back) * h, by - np.sin(back) * h], -1),
                                np.stack([tipx, tipy], -1),
                                np.stack([bx + np.cos(back) * h, by + np.sin(back) * h], -1)], 2))

    return dict(body=body, caudal=caudal, fins=np.stack(fins, 1), fin_on=np.stack(fin_on, 1),
                jaws=np.stack(jaws, 1), jaw_on=jaw > 0.12, spikes=np.stack(spikes, 2),
                spike_on=kk[None] < n_spk[:, None],
                fin_act=np.stack(fin_act, 1), stripes=stripes, stripe_on=j[None] < n_str[:, None],
                eyes=eyes, pupils=pupils, r_eye=r_eye, body_len=body_len,
                nose=np.stack([px[:, 0], py[:, 0]], -1))


def draw_creatures(surf, q, geo, ox, oy, scale):
    """creature_geometry の結果を画面に描く（ox, oy が q の原点、scale は 1 ワールド単位のピクセル数）"""
    n = len(q["dx"])
    if not n:
        return
    conv = lambda a: np.stack([ox + a[..., 0] * scale, oy - a[..., 1] * scale], -1).tolist()
    body, caudal, fins, stripes = conv(geo["body"]), conv(geo["caudal"]), conv(geo["fins"]), conv(geo["stripes"])
    eyes, pupils, jaws, spikes = conv(geo["eyes"]), conv(geo["pupils"]), conv(geo["jaws"]), conv(geo["spikes"])
    v = 0.62 + 0.38 * np.clip(q["energy"], 0, 1)  # お腹が空いているほど暗い
    hue, hue2 = q["hue"], q["hue"] + q["pattern_hue"]
    c_body = hsv_to_rgb(hue, np.full(n, 0.6), v).tolist()
    c_edge = hsv_to_rgb(hue, np.full(n, 0.7), v * 0.45).tolist()
    c_pat = hsv_to_rgb(hue2, np.full(n, 0.7), v * 0.6).tolist()
    c_tail = hsv_to_rgb(hue2, np.full(n, 0.45), v * 0.9).tolist()
    act = geo["fin_act"]
    c_fin = hsv_to_rgb(np.repeat(hue2[:, None], act.shape[1], 1), np.full(act.shape, 0.45),
                       np.minimum(v[:, None] * 0.85 + act * 0.3, 1)).tolist()
    # 瞳の色: 草食は黒、肉食ほど赤い
    d = np.clip(q["diet"], 0, 1)[:, None]
    c_pupil = (np.array([15, 15, 25]) * (1 - d) + np.array([220, 30, 30]) * d).astype(int).tolist()
    plen = geo["body_len"] * scale
    fin_on = geo["fin_on"] & (plen[:, None] * 0.1 > 0.6)
    fin_on_l, stripe_on, spike_on = fin_on.tolist(), geo["stripe_on"].tolist(), geo["spike_on"].tolist()
    jaw_on = (geo["jaw_on"] & (plen >= 12)).tolist()
    hurt = (q["hurt"] > 0).tolist()
    r_eye = (geo["r_eye"] * scale).tolist()
    detail, eyes_on, fine = (plen >= 16).tolist(), (plen >= 30).tolist(), (plen >= 45).tolist()
    stripe_w = np.maximum(1, 0.35 * q["size"] * scale).astype(int).tolist()
    poly, line, circle = pygame.draw.polygon, pygame.draw.line, pygame.draw.circle
    n_fins = act.shape[1]
    for i in range(n):
        poly(surf, c_tail[i], caudal[i])
        for k in range(n_fins):
            if fin_on_l[i][k]:
                poly(surf, c_fin[i][k], fins[i][k])
        if detail[i]:
            for k in range(6):
                if spike_on[i][k]:
                    poly(surf, c_edge[i], spikes[i][k][0])
                    poly(surf, c_edge[i], spikes[i][k][1])
        poly(surf, c_body[i], body[i])
        if jaw_on[i]:
            for j in range(2):
                poly(surf, (236, 226, 200), jaws[i][j])
                if fine[i]:
                    poly(surf, (90, 70, 60), jaws[i][j], 1)
        if detail[i]:
            for k in range(4):
                if stripe_on[i][k]:
                    line(surf, c_pat[i], stripes[i][k][0], stripes[i][k][1], stripe_w[i])
            if fine[i]:
                poly(surf, c_edge[i], body[i], 1)
                for k in range(n_fins):
                    if fin_on_l[i][k]:
                        poly(surf, c_edge[i], fins[i][k], 1)
        if hurt[i]:  # 噛まれた個体は赤く光る
            poly(surf, (255, 60, 60), body[i], 2)
        if eyes_on[i]:
            for e in range(2):
                circle(surf, (245, 245, 245), eyes[i][e], max(1.2, r_eye[i]))
                circle(surf, c_pupil[i], pupils[i][e], max(1, r_eye[i] * 0.5))


class Viewer:
    def __init__(self, world: World, args):
        self.world, self.args = world, args
        self.cfg = world.cfg
        self.tail_max = math.radians(self.cfg.tail_max_deg)
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
        self.info = None

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
        if d[i] < max(12, 4 * p["size"][i] * p["body_len"][i] * self.zoom):
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
        data, food, meat = self.world.render_data()
        # pygame は float32 を受け付けない
        data, food, self.meat = data.astype(np.float64), food.astype(np.float64), meat.astype(np.float64)
        p = {k: data[:, i] for i, k in enumerate(World.RENDER_COLS)}
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
                self.history.append((s["step"], s["food"], s.get("herbivores", 0), s.get("carnivores", 0)))
                self.history = self.history[-600:]
        now = time.time()
        if now - self._sps_t > 1.0:
            self.sps = (self.world.steps - self._sps_steps) / (now - self._sps_t)
            self._sps_t, self._sps_steps = now, self.world.steps
        self.info = self.world.creature_info(self.selected) if self.selected is not None else None
        if self.selected is not None and self.info is None:
            self.flash("the selected creature died")
            self.selected, self.follow = None, False

    def row(self, i, **override):
        """pack から 1 匹分を取り出す（描画関数にそのまま渡せる形）"""
        r = {k: v[i:i + 1].copy() for k, v in self.pack.items()}
        for k, v in override.items():
            r[k] = np.array([v], dtype=np.float64)
        return r

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

        # 肉（死体）: 大きい個体の死体ほど大きい
        if len(self.meat):
            mx, my = self.wrap_rel(self.meat[:, 0], self.meat[:, 1])
            sx, sy = self.to_screen(mx, my)
            for x, y, r in zip(sx.tolist(), sy.tolist(), (self.meat[:, 2] * self.zoom).tolist()):
                if -10 < x < VIEW_W + 10 and -10 < y < VIEW_H + 10:
                    pygame.draw.circle(self.screen, MEAT_EDGE, (x, y), max(2.0, r))
                    pygame.draw.circle(self.screen, MEAT, (x - r * 0.15, y - r * 0.15), max(1.5, r * 0.7))

        # 生き物
        if not len(p["slot"]):
            return
        sx, sy = self.to_screen(p["dx"], p["dy"])
        margin = 15 * self.zoom
        vis = np.nonzero((sx > -margin) & (sx < VIEW_W + margin) & (sy > -margin) & (sy < VIEW_H + margin))[0]
        if len(vis):
            q = {k: v[vis] for k, v in p.items()}
            draw_creatures(self.screen, q, creature_geometry(q, self.tail_max), VIEW_W / 2, VIEW_H / 2, self.zoom)
        if sel is not None:
            c = self.to_screen(p["dx"][sel], p["dy"][sel])
            r = 3.5 * p["size"][sel] * p["body_len"][sel] * self.zoom + 6
            pygame.draw.circle(self.screen, (255, 255, 255), c, r, 2)

    def draw_sensor(self, i):
        p, info, c = self.pack, self.info, self.cfg
        S = c.n_sectors
        th = p["theta"][i]
        nose = 0.43 * 6.0 * p["size"][i] * p["body_len"][i]
        hx, hy = p["dx"][i] + nose * math.cos(th), p["dy"][i] + nose * math.sin(th)
        fov, rng = math.radians(p["fov"][i]), p["sense_range"][i]
        food, meat, other = info["inputs"][:S], info["inputs"][S:2 * S], info["inputs"][2 * S:3 * S]
        hs = self.to_screen(hx, hy)
        wdt = fov / S
        for s in range(S):
            a0 = th - fov / 2 + s * wdt
            arc = [self.to_screen(hx + rng * math.cos(a0 + wdt * k / 4), hy + rng * math.sin(a0 + wdt * k / 4))
                   for k in range(5)]
            pygame.draw.polygon(self.overlay, (*SENSOR, 12 + int(140 * food[s])), [hs] + arc)
            if meat[s] > 0:
                d = rng * (1 - meat[s])
                am = a0 + wdt / 2
                pygame.draw.circle(self.overlay, (255, 80, 80, 220), self.to_screen(hx + d * math.cos(am), hy + d * math.sin(am)), 5, 2)
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
        self.y = 10
        spd = SPEEDS[self.speed_i]
        self.text("EVOLUTION", font=self.font_big, color=ACCENT)
        self.text(f"step {s['step']:,}   {'MAX' if spd == 0 else f'x{spd}'}   {self.sps:,.0f} st/s"
                  + ("   [PAUSED]" if self.paused else ""), color=DIM)
        self.text(f"population {s['population']:5d}", font=self.font_big)
        y = self.y
        self.text(f"herbivores {s.get('herbivores', 0):4d}", color=HERB_COL)
        self.y = y
        self.text(f"carnivores {s.get('carnivores', 0):4d}", color=CARN_COL, x=172)
        self.text(f"plants {s['food']}  corpses {s['meat']}  kills {s['kills']:,}")
        self.text(f"gen max {s.get('gen_max', 0)} (mean {s.get('gen_mean', 0):.0f})  families {s.get('lineages', 0)}")
        self.text(f"births {s['births']:,}  deaths {s['deaths']:,}  rescued {s['injected']}", color=DIM)
        self.y += 4
        self.draw_graph()
        self.y += 6
        self.draw_hue_strip()
        self.y += 4
        self.draw_portrait()
        self.y += 6
        if self.info is not None:
            self.draw_creature()
        else:
            self.draw_traits()
            self.y += 4
            self.draw_fin_map()
        yy = VIEW_H - 76
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
        self.y += surf.get_height() + 2

    def draw_graph(self):
        """植物・草食・肉食の数の移り変わり"""
        x, w, h = VIEW_W + 14, PANEL_W - 28, 70
        pygame.draw.rect(self.screen, BOX_BG, (x, self.y, w, h), border_radius=4)
        if len(self.history) > 1:
            hist = np.array(self.history, dtype=float)
            top = max(hist[:, 1:].max(), 1)
            t0, t1 = hist[0, 0], max(hist[-1, 0], hist[0, 0] + 1)
            for col, color in ((1, PLANT_COL), (2, HERB_COL), (3, CARN_COL)):
                pts = [(x + (t - t0) / (t1 - t0) * w, self.y + h - 4 - v / top * (h - 10))
                       for t, v in zip(hist[:, 0], hist[:, col])]
                pygame.draw.lines(self.screen, color, False, pts, 2)
            self.screen.blit(self.font.render(f"{int(top)}", True, DIM), (x + 4, self.y + 2))
        for label, color, dx in (("plants", PLANT_COL, 206), ("herb", HERB_COL, 146), ("carn", CARN_COL, 98)):
            self.screen.blit(self.font.render(label, True, color), (x + w - dx + 60, self.y + 2))
        self.y += h + 2

    def draw_hue_strip(self):
        hist = self.summary.get("hue_hist")
        x, w, h = VIEW_W + 14, PANEL_W - 28, 22
        self.text("body colors (family markers)", color=DIM)
        if hist:
            n, top = len(hist), max(max(hist), 1)
            cols = hsv_to_rgb((np.arange(n) + 0.5) / n, np.full(n, 0.62), np.full(n, 0.95)).tolist()
            bw = w / n
            for i, v in enumerate(hist):
                bh = int(h * (v / top) ** 0.5)
                if bh:
                    pygame.draw.rect(self.screen, cols[i], (x + i * bw, self.y + h - bh, bw - 1, bh))
        pygame.draw.line(self.screen, (60, 70, 90), (x, self.y + h), (x + w, self.y + h))
        self.y += h + 4

    def draw_portrait(self):
        """選択中の個体（いなければ子どもの一番多い個体）を大きく描く"""
        p = self.pack
        x, w, h = VIEW_W + 14, PANEL_W - 28, 104
        i = None
        if self.selected is not None:
            hit = np.nonzero(p["slot"] == self.selected)[0]
            i = int(hit[0]) if len(hit) else None
            title = "SELECTED"
        elif len(p["slot"]):
            i = int(np.argmax(p["children"]))
            title = f"TOP PARENT  ({int(p['children'][i])} children)"
        self.text(title if i is not None else "no creatures", color=ACCENT)
        box = pygame.Rect(x, self.y, w, h)
        pygame.draw.rect(self.screen, BOX_BG, box, border_radius=6)
        if i is not None:
            q = self.row(i, theta=0.0, dx=0.0, dy=0.0)
            geo = creature_geometry(q, self.tail_max)
            pts = np.concatenate([geo["body"][0], geo["caudal"][0], geo["fins"][0][geo["fin_on"][0]].reshape(-1, 2)])
            lo, hi = pts.min(0), pts.max(0)
            scale = min((w - 24) / max(hi[0] - lo[0], 1e-3), (h - 16) / max(hi[1] - lo[1], 1e-3))
            cx, cy = (lo + hi) / 2
            clip = self.screen.get_clip()
            self.screen.set_clip(box)
            draw_creatures(self.screen, q, geo, box.centerx - cx * scale, box.centery + cy * scale, scale)
            self.screen.set_clip(clip)
        self.y += h + 2

    def draw_traits(self):
        s = self.summary
        self.text("POPULATION AVERAGE  (bar = mean +- std)", color=ACCENT)
        if "trait_mean" not in s:
            return
        col_w = (PANEL_W - 28) // 2
        for k, name in enumerate(BODY_TRAITS):
            i = TRAIT_INDEX[name]
            lo, hi = TRAIT_RANGE[name]
            m, sd = s["trait_mean"][i], s["trait_std"][i]
            x = VIEW_W + 14 + (k % 2) * (col_w + 4)
            y = self.y + (k // 2) * 17
            self.screen.blit(self.font.render(TRAIT_LABELS[name], True, TEXT), (x, y))
            bx, bw, by = x + 84, col_w - 90, y + 6
            pygame.draw.rect(self.screen, (36, 44, 60), (bx, by, bw, 5), border_radius=3)
            a = (max(m - sd, lo) - lo) / (hi - lo)
            b = (min(m + sd, hi) - lo) / (hi - lo)
            pygame.draw.rect(self.screen, (70, 110, 150), (bx + a * bw, by, max(2, (b - a) * bw), 5), border_radius=3)
            pygame.draw.circle(self.screen, ACCENT, (bx + (m - lo) / (hi - lo) * bw, by + 2), 3)
        self.y += ((len(BODY_TRAITS) + 1) // 2) * 17 + 2

    def draw_fin_map(self):
        """集団全体のヒレの分布: 横=付いている位置（右が顔）、縦=向き（上が前向き）"""
        p = self.pack
        self.text("FIN MAP  (up: points fwd, down: back)", color=ACCENT)
        x, w, h = VIEW_W + 14, PANEL_W - 28, 74
        box = pygame.Rect(x, self.y, w, h)
        pygame.draw.rect(self.screen, BOX_BG, box, border_radius=4)
        for frac in (1 / 3, 2 / 3):
            pygame.draw.line(self.screen, (40, 50, 70), (x, self.y + h * frac), (x + w, self.y + h * frac))
        n = len(p["slot"])
        if n:
            idx = np.random.default_rng(0).permutation(n)[:250]
            for k in range(N_FIN_PAIRS):
                fs = p[f"fin{k}_size"][idx]
                on = fs > 0.02
                if not on.any():
                    continue
                fx = x + 6 + (1 - p[f"fin{k}_pos"][idx][on]) * (w - 12)
                fy = self.y + 4 + p[f"fin{k}_angle"][idx][on] / 180 * (h - 8)
                cols = hsv_to_rgb(p["hue"][idx][on] + p["pattern_hue"][idx][on], np.full(on.sum(), 0.45),
                                  np.full(on.sum(), 0.95)).tolist()
                for cx, cy, r, col in zip(fx.tolist(), fy.tolist(), (1 + fs[on] * 1.5).tolist(), cols):
                    pygame.draw.circle(self.screen, col, (cx, cy), r)
        self.screen.blit(self.font.render("face", True, DIM), (x + w - 34, self.y + h - 16))
        self.screen.blit(self.font.render("tail", True, DIM), (x + 4, self.y + h - 16))
        self.y += h + 2

    def draw_creature(self):
        info = self.info
        x, w = VIEW_W + 14, PANEL_W - 28
        tr = info["traits"]
        self.text(f"#{info['id']}  gen {info['generation']}  age {info['age']}  fam #{info['root']}"
                  + ("  [F]" if self.follow else ""))
        self.text(f"ate {info['food_eaten']:.0f}   children {info['children']}   kills {info['kills']}")
        for label, val, vmax, color in (("energy", info["energy"], info["energy_max"], (120, 230, 120)),
                                        ("health", info["hp"], info["hp_max"], (240, 90, 90))):
            self.screen.blit(self.font.render(label, True, DIM), (x, self.y))
            pygame.draw.rect(self.screen, (36, 44, 60), (x + 60, self.y + 5, w - 60, 7), border_radius=3)
            pygame.draw.rect(self.screen, color, (x + 60, self.y + 5, max(0, val / vmax) * (w - 60), 7), border_radius=3)
            self.y += 16
        self.text("FINS  where/points  size   use L  R", color=ACCENT)
        n_fin = 0
        for k in range(N_FIN_PAIRS):
            size = tr[TRAIT_INDEX[f"fin{k}_size"]]
            if size <= 0.02:
                continue
            n_fin += 1
            where, way = fin_words(tr[TRAIT_INDEX[f"fin{k}_pos"]], tr[TRAIT_INDEX[f"fin{k}_angle"]])
            al, ar = info["fin_act"][k], info["fin_act"][N_FIN_PAIRS + k]
            self.text(f" {k + 1}: {where:>4}/{way:<4}  {size:4.2f}   {al:.2f} {ar:.2f}")
        if not n_fin:
            self.text(" (no fins)", color=DIM)
        names = ["size", "body_len", "body_wid", "tail_power", "head_size", "jaw", "armor", "diet", "fov", "sense_range"]
        for a, b in zip(names[::2], names[1::2]):
            self.text(f"{TRAIT_LABELS[a]:<10}{tr[TRAIT_INDEX[a]]:6.2f}   {TRAIT_LABELS[b]:<10}{tr[TRAIT_INDEX[b]]:6.2f}",
                      color=DIM)
        S = self.cfg.n_sectors
        self.text("SEES  plants / meat / others", color=ACCENT)
        self.bars(info["inputs"][:S][::-1], SENSOR, 13)
        self.bars(info["inputs"][S:2 * S][::-1], (255, 90, 90), 13)
        self.bars(info["inputs"][2 * S:3 * S][::-1], OTHER, 13)
        biting = info["bite"] > 0.5
        self.text(f"bite urge {info['bite']:.2f}" + ("   BITING!" if biting else ""),
                  color=CARN_COL if biting else TEXT)

    def bars(self, values, color, h):
        x, w = VIEW_W + 14, PANEL_W - 28
        n = len(values)
        bw = w / n
        for i, v in enumerate(values):
            pygame.draw.rect(self.screen, (32, 38, 52), (x + i * bw + 1, self.y, bw - 2, h), border_radius=3)
            fh = int(h * v)
            if fh:
                pygame.draw.rect(self.screen, color, (x + i * bw + 1, self.y + h - fh, bw - 2, fh), border_radius=3)
        self.y += h + 3

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
    show = ["size", "head_size", "jaw", "armor", "diet", "tail_power", "sense_range"]
    for i in range(args.steps):
        world.step()
        if (i + 1) % 2000 == 0 or i + 1 == args.steps:
            s = world.summary()
            tm = s.get("trait_mean")
            traits = "  ".join(f"{TRAIT_LABELS[n]}={tm[TRAIT_INDEX[n]]:.2f}" for n in show) if tm else ""
            print(f"step {s['step']:7d}  {s['step'] / (time.time() - t0):5.0f} st/s  pop {s['population']:4d} "
                  f"(herb {s.get('herbivores', 0)} / carn {s.get('carnivores', 0)})  plants {s['food']:4d}  "
                  f"corpses {s['meat']:3d}  kills {s['kills']:5d}  gen {s.get('gen_max', 0):4d}  |  {traits}", flush=True)
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
