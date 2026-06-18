import os
import math
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import pygame
from pygame.locals import QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION
import tkinter as tk
from tkinter import filedialog
from matplotlib import cm

from getFeatures import extract_frame_features
from model import LSTMEncoder


# ===================== 設定 =====================
MODEL_PATH       = ""
FRAME_LENGTH_SEC = 0.25
HOP_LENGTH_SEC   = FRAME_LENGTH_SEC

RESULT_ROOT = "pygame_interactive_results"

EMB_CACHE_DIR = Path("embeddings_cache")
EMB_CACHE_DIR.mkdir(exist_ok=True)

SCREEN_WIDTH  = 1200
SCREEN_HEIGHT = 600

HEATMAP_MARGIN = 20
HEATMAP_WIDTH  = 900
HEATMAP_HEIGHT = 550

BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

BTN_BG = (230, 230, 230)
BTN_BORDER = (0, 0, 0)
BTN_TEXT = (0, 0, 0)

SELECTION_FILL = (0, 0, 0, 35)
SELECTION_BORDER = (0, 0, 0)

PLAY_LINE_COLOR = (255, 0, 0)
PLAY_LINE_W = 2


# ===================== pygame 初期化 =====================
pygame.init()
pygame.mixer.init()
pygame.font.init()

FONT_PATH = "fonts\ipaexg.ttf"
FONT_SIZE = 17
try:
    FONT = pygame.font.Font(FONT_PATH, FONT_SIZE)
except FileNotFoundError:
    print(f"[Warn] Japanese font not found: {FONT_PATH} -> fallback")
    FONT = pygame.font.SysFont(None, FONT_SIZE)

FONT_SMALL = pygame.font.SysFont(None, 18)
FONT_BIG   = pygame.font.SysFont(None, 24)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_text_to_width(text, font, max_width):
    if font.size(text)[0] <= max_width:
        return text
    ellipsis = "…"
    max_width -= font.size(ellipsis)[0]
    trimmed = text
    while trimmed and font.size(trimmed)[0] > max_width:
        trimmed = trimmed[:-1]
    return trimmed + ellipsis

def extract_song_title_from_wav(wav_path: str):
    # 歌手名_曲名_audio.wav -> 曲名
    stem = Path(wav_path).stem
    stem = stem.replace("_audio", "")
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem

def choose_wav_file_dialog(title="Select WAV file"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    root.destroy()
    return path

def cache_path_for(wav_path: str) -> Path:
    base = Path(wav_path).stem
    return EMB_CACHE_DIR / f"{base}.csv"

def load_embeddings_csv(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        arr = np.loadtxt(csv_path, delimiter=",")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr
    except Exception as e:
        print(f"[Warn] Failed to load cache {csv_path}: {e}")
        return None

def save_embeddings_csv(csv_path: Path, embeddings: np.ndarray):
    np.savetxt(csv_path, embeddings, delimiter=",")
    csv_path.touch(exist_ok=True)

def extract_embeddings(wav_path: str, model, frame_length_sec: float):
    cache_path = cache_path_for(wav_path)
    emb = load_embeddings_csv(cache_path)
    if emb is not None:
        print(f"[Cache] loaded: {cache_path}")
        return emb

    print(f"[Feat] Extracting features from {wav_path} ...")
    feat = extract_frame_features(
        wav_path,
        sr=44100,
        frame_length_sec=frame_length_sec,
        normalize=None
    )["frame_vectors"]

    feats_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        embeddings = model(feats_tensor).cpu().numpy()

    save_embeddings_csv(cache_path, embeddings)
    print(f"[Cache] saved: {cache_path}")
    return embeddings

def compute_similarity_matrix(emb1: np.ndarray, emb2: np.ndarray):
    norm1 = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    norm2 = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    return np.dot(norm1, norm2.T)  # (T1, T2)

def create_heatmap_surface(sim_matrix: np.ndarray):
    vmin = sim_matrix.min()
    vmax = sim_matrix.max()
    if vmax - vmin < 1e-8:
        norm = np.zeros_like(sim_matrix)
    else:
        norm = (sim_matrix - vmin) / (vmax - vmin)

    cmap = cm.get_cmap("viridis")
    rgba = cmap(norm)[:, :, :3]
    rgb = (rgba * 255).astype(np.uint8)

    rgb = np.transpose(rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(rgb)

def cut_and_save_pair(
    wav1_path: str,
    wav2_path: str,
    start1_sec: float,
    end1_sec: float,
    start2_sec: float,
    end2_sec: float,
    out_root: str,
    pair_index: int,
    min_dur_sec: float = 0.25,
):
    os.makedirs(out_root, exist_ok=True)

    a1, sr1 = sf.read(wav1_path, always_2d=False)
    a2, sr2 = sf.read(wav2_path, always_2d=False)
    len1_sec = len(a1) / sr1
    len2_sec = len(a2) / sr2

    s1 = max(0.0, min(start1_sec, end1_sec))
    e1 = min(len1_sec, max(start1_sec, end1_sec))
    s2 = max(0.0, min(start2_sec, end2_sec))
    e2 = min(len2_sec, max(start2_sec, end2_sec))

    if (e1 - s1) < min_dur_sec or (e2 - s2) < min_dur_sec:
        print(f"[Skip] too short: song1={e1 - s1:.3f}s, song2={e2 - s2:.3f}s")
        return None, None

    s1_samp = int(round(s1 * sr1))
    e1_samp = int(round(e1 * sr1))
    s2_samp = int(round(s2 * sr2))
    e2_samp = int(round(e2 * sr2))

    name1 = Path(wav1_path).stem
    name2 = Path(wav2_path).stem

    sub = os.path.join(out_root, f"pair_{pair_index:03d}_{s1:.2f}-{e1:.2f}_{s2:.2f}-{e2:.2f}")
    os.makedirs(sub, exist_ok=True)

    out1 = os.path.join(sub, f"{name1}_A_pair{pair_index:03d}.wav")
    out2 = os.path.join(sub, f"{name2}_B_pair{pair_index:03d}.wav")

    sf.write(out1, a1[s1_samp:e1_samp], sr1, subtype="PCM_16")
    sf.write(out2, a2[s2_samp:e2_samp], sr2, subtype="PCM_16")

    print(f"[Saved] {out1}")
    print(f"[Saved] {out2}")
    return out1, out2

# ===================== VUメーター（アナログ） =====================
def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def _lerp(a, b, t):
    return a + (b - a) * t

def _polar_point(cx, cy, r, a):
    return (cx + r * math.cos(a), cy + r * math.sin(a))

def _draw_arc(screen, color, cx, cy, r, a0, a1, width=2, steps=160):
    if a1 < a0:
        a0, a1 = a1, a0
    pts = []
    for i in range(steps + 1):
        a = a0 + (a1 - a0) * (i / steps)
        pts.append(_polar_point(cx, cy, r, a))
    pygame.draw.lines(screen, color, False, pts, width)

def _exp_map(x, k=3.0):
    x = _clamp(x, 0.0, 1.0)
    return (math.exp(k * x) - 1.0) / (math.exp(k) - 1.0)

class VUMeterAnalog:

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.val_min = -1.0
        self.val_max =  1.0
        self.value = self.val_min
        self.peak  = self.val_min

        self.a0 = math.radians(210)
        self.a1 = math.radians(330)

        self.tick_exp_k = 3.0
        self.attack  = 0.35
        self.release = 0.06

        self.red_from = 0.7
        self.red_to   = 1.0

    def _angle_from_value(self, v):
        v = _clamp(v, self.val_min, self.val_max)
        t = (v - self.val_min) / (self.val_max - self.val_min)
        t = _exp_map(t, self.tick_exp_k)
        return _lerp(self.a0, self.a1, t)

    def update(self, value, dt):
        if value is None:
            v = self.val_min
        else:
            v = _clamp(float(value), self.val_min, self.val_max)

        if v > self.value:
            k = 1.0 - (1.0 - self.attack) ** (dt * 60.0)
        else:
            k = 1.0 - (1.0 - self.release) ** (dt * 60.0)

        self.value = _lerp(self.value, v, k)

        if value is None:
            self.peak = _lerp(self.peak, self.val_min, 0.25)
        else:
            if v > self.peak:
                self.peak = v
            else:
                self.peak = _lerp(self.peak, self.val_min, 0.02)

    def draw(self, screen, font_big, font_small):
        prev_clip = screen.get_clip()
        screen.set_clip(self.rect)

        pygame.draw.rect(screen, (20, 20, 20), self.rect, border_radius=8)
        inner = self.rect.inflate(-14, -14)
        pygame.draw.rect(screen, (50, 50, 50), inner, border_radius=6)

        panel = inner.inflate(-18, -18)
        pygame.draw.rect(screen, (235, 220, 150), panel, border_radius=4)

        cx = panel.centerx
        cy = panel.bottom - int(panel.height * 0.09)
        r = int(min(panel.width * 0.48, panel.height * 0.80))

        _draw_arc(screen, (30, 30, 30), cx, cy, r, self.a0, self.a1, 3)

        for i in range(11):
            t = i / 10.0
            v = _lerp(self.val_min, self.val_max, t)
            a = self._angle_from_value(v)

            p0 = _polar_point(cx, cy, r - 6, a)
            p1 = _polar_point(cx, cy, r - 18, a)
            col = (160, 0, 0) if v >= self.red_from else (40, 40, 40)
            pygame.draw.line(screen, col, p0, p1, 2)

            if i % 2 == 0:
                label = f"{v:+.1f}"
                ts = font_small.render(label, True, (25, 25, 25))
                tx, ty = _polar_point(cx, cy, r - 34, a)
                screen.blit(ts, ts.get_rect(center=(int(tx), int(ty))))

        a_val = self._angle_from_value(self.value)
        p_needle = _polar_point(cx, cy, r - 22, a_val)
        pygame.draw.line(screen, (160, 0, 0), (cx, cy), p_needle, 4)
        pygame.draw.circle(screen, (40, 40, 40), (cx, cy), 8)

        #a_peak = self._angle_from_value(self.peak)
        #p_peak = _polar_point(cx, cy, r - 22, a_peak)
        #pygame.draw.circle(screen, (20, 20, 20), (int(p_peak[0]), int(p_peak[1])), 4)

        title = font_big.render("similarity", True, (25, 25, 25))
        screen.blit(title, (panel.left + 8, panel.top + 6))

        valtxt = font_small.render(f"{self.value:+.3f}", True, (25, 25, 25))
        screen.blit(valtxt, (panel.left + 8, panel.bottom - panel.height*0.1))

        screen.set_clip(prev_clip)


# ===================== ボタン =====================
class Button:
    def __init__(self, rect, text, callback, bg_color=BTN_BG, text_color=BTN_TEXT):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.bg_color = bg_color
        self.text_color = text_color

    def draw(self, screen):
        pygame.draw.rect(screen, self.bg_color, self.rect)
        pygame.draw.rect(screen, BTN_BORDER, self.rect, 2)

        padding = 10
        max_text_width = self.rect.width - padding * 2
        safe_text = fit_text_to_width(self.text, FONT, max_text_width)

        text_surf = FONT.render(safe_text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()


def diag_similarity_at_playhead(sim_matrix, last_selected_raw, play_mode, cur_axis_sec, last_cut_times):
    if sim_matrix is None or last_selected_raw is None or last_cut_times is None:
        return None

    x0, y0, x1, y1 = last_selected_raw
    w = (x1 - x0)
    h = (y1 - y0)
    if abs(w) < 1e-8 or abs(h) < 1e-8:
        return None

    if play_mode == 'A':
        s_sec = last_cut_times['s1']
        e_sec = last_cut_times['e1']
    elif play_mode == 'B':
        s_sec = last_cut_times['s2']
        e_sec = last_cut_times['e2']
    else:
        return None

    dur = max(e_sec - s_sec, 1e-8)
    rel = (cur_axis_sec - s_sec) / dur
    rel = max(0.0, min(1.0, rel))

    raw_x = x0 + rel * w
    raw_y = y0 + rel * h

    row = int(raw_y)
    col = int(raw_x)

    T1, T2 = sim_matrix.shape
    row = max(0, min(row, T1 - 1))
    col = max(0, min(col, T2 - 1))
    return float(sim_matrix[row, col])


# ===================== main =====================
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("")

    heat_rect = pygame.Rect(HEATMAP_MARGIN, HEATMAP_MARGIN, HEATMAP_WIDTH, HEATMAP_HEIGHT)

    wav1_path = None
    wav2_path = None
    name1 = "(未選択)"
    name2 = "(未選択)"

    model = None
    heatmap_surface_raw = None
    sim_matrix = None
    T1 = T2 = None

    zoom = 1.0
    ZOOM_MIN = 0.25
    ZOOM_MAX = 8.0
    offset_x = 0.0
    offset_y = 0.0
    panning = False
    pan_last = None

    scaled_heatmap_cache = None
    scaled_zoom_cache = None
    scaled_w_cache = None
    scaled_h_cache = None

    # 再生状態
    play_mode = None
    play_start_ticks = None
    play_start_sec = None
    play_duration_sec = None

    frozen_line_mode = None      
    frozen_line_axis_sec = None  
    frozen_line_active = False   

    last_cut_times = None
    last_cut_1 = None
    last_cut_2 = None

    selecting = False
    select_start_raw = None
    select_end_raw = None
    last_selected_raw = None
    selection_index = 1

    result_root = os.path.join(RESULT_ROOT, "manual_pairs")
    os.makedirs(result_root, exist_ok=True)

    btn_width = SCREEN_WIDTH - (HEATMAP_MARGIN * 2 + HEATMAP_WIDTH) - 40
    btn_x = HEATMAP_MARGIN + HEATMAP_WIDTH + 20
    btn_y = HEATMAP_MARGIN + 20
    btn_h = 40
    gap = 12

    buttons = []


    vu_top = btn_y + (btn_h + gap) * 5 + 10
    vu_h = 160
    vu_rect = pygame.Rect(btn_x, vu_top, btn_width, vu_h)
    vu_meter = VUMeterAnalog(vu_rect)

    def clamp_offsets():
        nonlocal offset_x, offset_y
        if scaled_w_cache is None or scaled_h_cache is None:
            offset_x = 0.0
            offset_y = 0.0
            return
        max_off_x = max(0.0, float(scaled_w_cache - heat_rect.width))
        max_off_y = max(0.0, float(scaled_h_cache - heat_rect.height))
        offset_x = max(0.0, min(offset_x, max_off_x))
        offset_y = max(0.0, min(offset_y, max_off_y))

    def ensure_scaled_heatmap():
        nonlocal scaled_heatmap_cache, scaled_zoom_cache, scaled_w_cache, scaled_h_cache
        if heatmap_surface_raw is None:
            scaled_heatmap_cache = None
            scaled_zoom_cache = None
            scaled_w_cache = None
            scaled_h_cache = None
            return None
        if scaled_heatmap_cache is not None and scaled_zoom_cache == zoom:
            return scaled_heatmap_cache

        raw_w, raw_h = heatmap_surface_raw.get_size()
        scaled_w = max(1, int(raw_w * zoom))
        scaled_h = max(1, int(raw_h * zoom))

        scaled_heatmap_cache = pygame.transform.smoothscale(heatmap_surface_raw, (scaled_w, scaled_h))
        scaled_zoom_cache = zoom
        scaled_w_cache = scaled_w
        scaled_h_cache = scaled_h
        clamp_offsets()
        return scaled_heatmap_cache

    def screen_to_raw(mx, my):
        if heatmap_surface_raw is None:
            return None
        if not heat_rect.collidepoint((mx, my)):
            return None
        sx = mx - heat_rect.left
        sy = my - heat_rect.top
        raw_x = (offset_x + sx) / zoom
        raw_y = (offset_y + sy) / zoom
        raw_w, raw_h = heatmap_surface_raw.get_size()
        raw_x = max(0.0, min(raw_x, raw_w - 1e-6))
        raw_y = max(0.0, min(raw_y, raw_h - 1e-6))
        return (raw_x, raw_y)

    def raw_rect_to_screen_rect(x0, y0, x1, y1):
        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = abs(x1 - x0)
        rh = abs(y1 - y0)
        x = heat_rect.left + (rx * zoom - offset_x)
        y = heat_rect.top  + (ry * zoom - offset_y)
        w = rw * zoom
        h = rh * zoom
        return pygame.Rect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def zoom_at(mx, my, zoom_factor):
        nonlocal zoom, offset_x, offset_y, scaled_heatmap_cache, scaled_zoom_cache
        pt = screen_to_raw(mx, my)
        if pt is None:
            return
        raw_x, raw_y = pt
        new_zoom = zoom * zoom_factor
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, new_zoom))
        if abs(new_zoom - zoom) < 1e-12:
            return

        sx = mx - heat_rect.left
        sy = my - heat_rect.top

        zoom = new_zoom
        scaled_heatmap_cache = None
        scaled_zoom_cache = None
        ensure_scaled_heatmap()

        offset_x = raw_x * zoom - sx
        offset_y = raw_y * zoom - sy
        clamp_offsets()

    def reset_view_states():
        nonlocal zoom, offset_x, offset_y, panning, pan_last
        nonlocal scaled_heatmap_cache, scaled_zoom_cache, scaled_w_cache, scaled_h_cache
        zoom = 1.0
        offset_x = 0.0
        offset_y = 0.0
        panning = False
        pan_last = None
        scaled_heatmap_cache = None
        scaled_zoom_cache = None
        scaled_w_cache = None
        scaled_h_cache = None

    def ensure_model_loaded():
        nonlocal model
        if model is not None:
            return True
        if not os.path.exists(MODEL_PATH):
            print(f"[Error] MODEL_PATH not found: {MODEL_PATH}")
            return False
        print("[Model] Loading LSTMEncoder ...")
        model = LSTMEncoder(input_dim=53, hidden_dim=128, output_dim=32).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return True

    def clear_frozen_line():
        nonlocal frozen_line_active, frozen_line_mode, frozen_line_axis_sec
        frozen_line_active = False
        frozen_line_mode = None
        frozen_line_axis_sec = None

    def recompute_heatmap_if_ready():
        nonlocal heatmap_surface_raw, sim_matrix, T1, T2
        nonlocal selecting, select_start_raw, select_end_raw, last_selected_raw
        nonlocal selection_index, last_cut_times, last_cut_1, last_cut_2
        nonlocal play_mode, play_start_ticks, play_start_sec, play_duration_sec
        nonlocal frozen_line_active, frozen_line_mode, frozen_line_axis_sec

        if not wav1_path or not wav2_path:
            return
        if not ensure_model_loaded():
            return

        selecting = False
        select_start_raw = None
        select_end_raw = None
        last_selected_raw = None
        selection_index = 1
        last_cut_times = None
        last_cut_1 = None
        last_cut_2 = None

        pygame.mixer.music.stop()
        play_mode = None
        play_start_ticks = None
        play_start_sec = None
        play_duration_sec = None
        clear_frozen_line()

        emb1 = extract_embeddings(wav1_path, model, FRAME_LENGTH_SEC)
        emb2 = extract_embeddings(wav2_path, model, FRAME_LENGTH_SEC)
        sim_matrix = compute_similarity_matrix(emb1, emb2)
        T1, T2 = sim_matrix.shape
        print(f"[Sim] matrix shape = {sim_matrix.shape}")

        heatmap_surface_raw = create_heatmap_surface(sim_matrix)
        reset_view_states()

    def start_play(mode: str, wav_path: str, axis_start_sec: float, axis_end_sec: float):
        nonlocal play_mode, play_start_ticks, play_start_sec, play_duration_sec
        clear_frozen_line()

        if wav_path and os.path.exists(wav_path):
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()
            play_mode = mode
            play_start_ticks = pygame.time.get_ticks()
            play_start_sec = axis_start_sec
            play_duration_sec = max(0.0, axis_end_sec - axis_start_sec)
            print(f"[Play] {mode}: {wav_path}")
        else:
            print(f"[Play] {mode}: まだ切り出しがありません")
            play_mode = None
            play_start_ticks = None
            play_start_sec = None
            play_duration_sec = None

    def play_last_cut_1():
        nonlocal last_cut_1, last_cut_times
        if last_cut_times is None:
            print("[Play] A: まだ切り出しがありません")
            return
        start_play('A', last_cut_1, last_cut_times['s1'], last_cut_times['e1'])

    def play_last_cut_2():
        nonlocal last_cut_2, last_cut_times
        if last_cut_times is None:
            print("[Play] B: まだ切り出しがありません")
            return
        start_play('B', last_cut_2, last_cut_times['s2'], last_cut_times['e2'])

    def stop_playback():
        nonlocal play_mode, play_start_ticks, play_start_sec, play_duration_sec
        pygame.mixer.music.stop()
        play_mode = None
        play_start_ticks = None
        play_start_sec = None
        play_duration_sec = None
        clear_frozen_line()
        print("[Stop]")

    def on_select_a():
        nonlocal wav1_path, name1
        path = choose_wav_file_dialog("Select WAV for A")
        if path:
            wav1_path = path
            name1 = extract_song_title_from_wav(wav1_path)
            if wav2_path:
                recompute_heatmap_if_ready()

    def on_select_b():
        nonlocal wav2_path, name2
        path = choose_wav_file_dialog("Select WAV for B")
        if path:
            wav2_path = path
            name2 = extract_song_title_from_wav(wav2_path)
            if wav1_path:
                recompute_heatmap_if_ready()

    buttons.append(Button((btn_x, btn_y + (btn_h + gap) * 0, btn_width, btn_h), "Select WAV A", on_select_a))
    buttons.append(Button((btn_x, btn_y + (btn_h + gap) * 1, btn_width, btn_h), "Select WAV B", on_select_b))
    buttons.append(Button((btn_x, btn_y + (btn_h + gap) * 2, btn_width, btn_h), f"Play A : {name1}", play_last_cut_1))
    buttons.append(Button((btn_x, btn_y + (btn_h + gap) * 3, btn_width, btn_h), f"Play B : {name2}", play_last_cut_2))
    buttons.append(Button((btn_x, btn_y + (btn_h + gap) * 4, btn_width, btn_h), "Stop", stop_playback))

    def refresh_play_button_labels():
        buttons[2].text = f"Play A : {name1}"
        buttons[3].text = f"Play B : {name2}"

    def freeze_line_at_end_of_play():
        nonlocal frozen_line_active, frozen_line_mode, frozen_line_axis_sec
        nonlocal play_mode, play_start_sec, play_duration_sec
        if play_mode is None or play_start_sec is None or play_duration_sec is None:
            return
        frozen_line_mode = play_mode
        frozen_line_axis_sec = play_start_sec + play_duration_sec 
        frozen_line_active = True

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            for b in buttons:
                b.handle_event(event)

            if heatmap_surface_raw is not None and sim_matrix is not None:
                if event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if heat_rect.collidepoint((mx, my)):
                        if event.y > 0:
                            zoom_at(mx, my, 1.15)
                        elif event.y < 0:
                            zoom_at(mx, my, 1 / 1.15)

                if event.type == MOUSEBUTTONDOWN:
                    if heat_rect.collidepoint(event.pos):
                        if event.button == 3 or event.button == 2:
                            panning = True
                            pan_last = event.pos
                        if event.button == 1:
                            selecting = True
                            select_start_raw = screen_to_raw(*event.pos)
                            select_end_raw = select_start_raw

                elif event.type == MOUSEMOTION:
                    if panning and pan_last is not None:
                        mx, my = event.pos
                        lx, ly = pan_last
                        dx = mx - lx
                        dy = my - ly
                        offset_x -= dx
                        offset_y -= dy
                        pan_last = (mx, my)
                        clamp_offsets()
                    if selecting:
                        select_end_raw = screen_to_raw(*event.pos)

                elif event.type == MOUSEBUTTONUP:
                    if event.button == 3 or event.button == 2:
                        panning = False
                        pan_last = None

                    if event.button == 1 and selecting:
                        selecting = False
                        if select_start_raw and select_end_raw:
                            sx, sy = select_start_raw
                            ex, ey = select_end_raw
                            x0 = min(sx, ex)
                            y0 = min(sy, ey)
                            x1 = max(sx, ex)
                            y1 = max(sy, ey)

                            if (x1 - x0) > 3.0 and (y1 - y0) > 3.0:
                                last_selected_raw = (x0, y0, x1, y1)

                                start_col = int(np.floor(x0))
                                end_col   = int(np.floor(x1))
                                start_row = int(np.floor(y0))
                                end_row   = int(np.floor(y1))

                                start_col = max(0, min(start_col, T2 - 1))
                                end_col   = max(0, min(end_col,   T2 - 1))
                                start_row = max(0, min(start_row, T1 - 1))
                                end_row   = max(0, min(end_row,   T1 - 1))

                                start2_sec = start_col * HOP_LENGTH_SEC
                                end2_sec   = (end_col + 1) * HOP_LENGTH_SEC
                                start1_sec = start_row * HOP_LENGTH_SEC
                                end1_sec   = (end_row + 1) * HOP_LENGTH_SEC

                                out1, out2 = cut_and_save_pair(
                                    wav1_path=wav1_path,
                                    wav2_path=wav2_path,
                                    start1_sec=start1_sec,
                                    end1_sec=end1_sec,
                                    start2_sec=start2_sec,
                                    end2_sec=end2_sec,
                                    out_root=result_root,
                                    pair_index=selection_index,
                                    min_dur_sec=0.25
                                )
                                if out1 and out2:
                                    last_cut_1 = out1
                                    last_cut_2 = out2
                                    selection_index += 1
                                    last_cut_times = {
                                        's1': float(start1_sec),
                                        'e1': float(end1_sec),
                                        's2': float(start2_sec),
                                        'e2': float(end2_sec),
                                    }

                        select_start_raw = None
                        select_end_raw = None

        # ---- 再生終了検出：ここで赤ラインを最後位置に固定してから再生状態をクリア ----
        if play_mode is not None and not pygame.mixer.music.get_busy():
            freeze_line_at_end_of_play()

            # 再生状態はクリア（でも frozen_line_* は残る）
            play_mode = None
            play_start_ticks = None
            play_start_sec = None
            play_duration_sec = None

        refresh_play_button_labels()

        cur_axis_sec = None
        cur_line_mode = None

        if play_mode is not None and play_start_ticks is not None and play_duration_sec is not None:
            elapsed = (pygame.time.get_ticks() - play_start_ticks) / 1000.0
            frac = max(0.0, min(1.0, elapsed / max(play_duration_sec, 1e-8)))
            cur_axis_sec = play_start_sec + frac * play_duration_sec
            cur_line_mode = play_mode
        elif frozen_line_active and frozen_line_axis_sec is not None and frozen_line_mode is not None:
            cur_axis_sec = frozen_line_axis_sec
            cur_line_mode = frozen_line_mode

        vu_val = None
        if (
            play_mode is not None
            and cur_axis_sec is not None
            and sim_matrix is not None
            and last_selected_raw is not None
            and last_cut_times is not None
        ):
            vu_val = diag_similarity_at_playhead(sim_matrix, last_selected_raw, play_mode, cur_axis_sec, last_cut_times)
        vu_meter.update(vu_val, dt)

        # ---------------- draw ----------------
        screen.fill(BG_COLOR)

        if heatmap_surface_raw is not None:
            scaled = ensure_scaled_heatmap()
            sw, sh = scaled.get_width(), scaled.get_height()

            sx = int(max(0, offset_x))
            sy = int(max(0, offset_y))
            blit_w = max(0, min(heat_rect.width,  sw - sx))
            blit_h = max(0, min(heat_rect.height, sh - sy))

            if blit_w > 0 and blit_h > 0:
                src = pygame.Rect(sx, sy, blit_w, blit_h)
                screen.blit(scaled, heat_rect.topleft, area=src)
            pygame.draw.rect(screen, (0, 0, 0), heat_rect, 1)
        else:
            msg1 = "右のボタンから WAV A / WAV B を選択してください"
            msg2 = "2曲揃うと自動で類似度マップを作成します"
            screen.blit(FONT.render(msg1, True, TEXT_COLOR), (heat_rect.left + 20, heat_rect.top + 20))
            screen.blit(FONT.render(msg2, True, TEXT_COLOR), (heat_rect.left + 20, heat_rect.top + 55))
            pygame.draw.rect(screen, (0, 0, 0), heat_rect, 1)

        rect_to_draw = None
        if heatmap_surface_raw is not None:
            if selecting and select_start_raw and select_end_raw:
                sx, sy = select_start_raw
                ex, ey = select_end_raw
                rect_to_draw = raw_rect_to_screen_rect(sx, sy, ex, ey)
            elif last_selected_raw is not None:
                x0, y0, x1, y1 = last_selected_raw
                rect_to_draw = raw_rect_to_screen_rect(x0, y0, x1, y1)

        if rect_to_draw and rect_to_draw.width > 0 and rect_to_draw.height > 0:
            clipped = rect_to_draw.clip(heat_rect)
            if clipped.width > 0 and clipped.height > 0:
                s = pygame.Surface(clipped.size, pygame.SRCALPHA)
                s.fill(SELECTION_FILL)
                screen.blit(s, clipped.topleft)
                pygame.draw.rect(screen, SELECTION_BORDER, clipped, 1)


        if (
            cur_axis_sec is not None
            and cur_line_mode is not None
            and last_selected_raw is not None
            and last_cut_times is not None
        ):
            x0, y0, x1, y1 = last_selected_raw
            sel_screen_rect = raw_rect_to_screen_rect(x0, y0, x1, y1).clip(heat_rect)

            if sel_screen_rect.width > 0 and sel_screen_rect.height > 0:
                if cur_line_mode == 'A':
                    s_sec = last_cut_times['s1']
                    e_sec = last_cut_times['e1']
                    if e_sec > s_sec:
                        rel = (cur_axis_sec - s_sec) / (e_sec - s_sec)
                        rel = max(0.0, min(1.0, rel))
                        y = int(sel_screen_rect.top + rel * sel_screen_rect.height)
                        pygame.draw.line(
                            screen, PLAY_LINE_COLOR,
                            (sel_screen_rect.left, y),
                            (sel_screen_rect.right, y),
                            PLAY_LINE_W
                        )

                elif cur_line_mode == 'B':
                    s_sec = last_cut_times['s2']
                    e_sec = last_cut_times['e2']
                    if e_sec > s_sec:
                        rel = (cur_axis_sec - s_sec) / (e_sec - s_sec)
                        rel = max(0.0, min(1.0, rel))
                        x = int(sel_screen_rect.left + rel * sel_screen_rect.width)
                        pygame.draw.line(
                            screen, PLAY_LINE_COLOR,
                            (x, sel_screen_rect.top),
                            (x, sel_screen_rect.bottom),
                            PLAY_LINE_W
                        )

        top_info = f"Heatmap: {name1} (Y) vs {name2} (X)"
        screen.blit(FONT.render(top_info, True, TEXT_COLOR), (HEATMAP_MARGIN, HEATMAP_MARGIN - 20))
        screen.blit(FONT.render(f"zoom: {zoom:.2f}", True, TEXT_COLOR), (heat_rect.right - 140, heat_rect.top - 20))

        for b in buttons:
            b.draw(screen)

        vu_meter.draw(screen, FONT_BIG, FONT_SMALL)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()


