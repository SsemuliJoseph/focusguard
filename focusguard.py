#!/usr/bin/env python3
"""
FocusGuard — Live Classical Computer Vision Demo
================================================

A real-time webcam demo that makes classical CV mathematics visible at
the pixel level. Two windows open side by side:

  * FocusGuard  — the mood/focus detector (product)
  * Math Panel  — the arithmetic the detector is running (glass box)

Designed for classroom projection: 22px+ body text, 28px+ cell values,
color-coded panels, on-screen legend.

Controls
--------
    1/2/3/4   Switch math panel (Kernel / Morphology / Otsu / Hough)
    k         Cycle kernel
    m         Cycle structuring element
    o         Toggle dilate <-> erode
    l         Toggle LAB <-> DEMO
    s         Toggle liveness
    q         Quit (runs session report)

Requires: opencv-contrib-python, numpy, matplotlib
"""

#!/usr/bin/env python3
"""
FocusGuard — Live Classical Computer Vision Demo

Two windows side by side:
  * FocusGuard  — mood/focus detector (readable by non-experts)
  * Math Panel  — the arithmetic it runs (glass box)

Controls: 1/2/3/4 panel | k kernel | m SE | o dil/ero | l LAB/DEMO | s live | q quit
"""

import os, time, shutil, subprocess, collections
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# CONSTANTS
# ============================================================================
WINDOW_MAIN = "FocusGuard"
WINDOW_MATH = "Math Panel"

CAM_INDEX = 0
WORK_W, WORK_H = 480, 360
DETECT_EVERY = 5
MATH_REDRAW_EVERY = 8

# Colors (BGR)
C_BG       = (24, 20, 18)
C_PANEL    = (36, 32, 30)
C_PANEL_HI = (56, 48, 44)
C_BORDER   = (110, 90, 90)
C_TEXT     = (240, 232, 232)
C_MUTED    = (168, 148, 148)
C_POS      = (160, 255, 125)
C_NEG      = (107, 107, 255)
C_ZERO     = (140, 140, 140)
C_AMBER    = (102, 209, 255)
C_PURPLE   = (255, 125, 199)
C_BLUE     = (222, 168, 78)
C_GREEN    = (160, 255, 125)
C_RED      = (107, 107, 255)
C_YELLOW   = (0, 220, 220)
C_DIM      = (70, 70, 90)

F_TITLE = cv2.FONT_HERSHEY_TRIPLEX
F_BODY  = cv2.FONT_HERSHEY_SIMPLEX

KERNELS = {
    "sobel_x":   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y":   np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "prewitt_x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "prewitt_y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "roberts_x": np.array([[1, 0], [0, -1]], dtype=np.float32),
    "roberts_y": np.array([[0, 1], [-1, 0]], dtype=np.float32),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "gaussian":  np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0,
    "box_blur":  np.ones((3, 3), dtype=np.float32) / 9.0,
}
KERNEL_ORDER = list(KERNELS.keys())

SE_KERNELS = {
    "square_3":  cv2.getStructuringElement(cv2.MORPH_RECT,    (3, 3)),
    "cross_3":   cv2.getStructuringElement(cv2.MORPH_CROSS,   (3, 3)),
    "ellipse_5": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    "hline_1x9": cv2.getStructuringElement(cv2.MORPH_RECT,    (9, 1)),
    "vline_9x1": cv2.getStructuringElement(cv2.MORPH_RECT,    (1, 9)),
    "diag_5":    np.eye(5, dtype=np.uint8),
}
SE_ORDER = list(SE_KERNELS.keys())
MORPH_OPS = ["dilate", "erode"]

# AU presentation metadata — friendly name, technical name, hint
AU_INFO = [
    # (short_label, plain_name, technical_name, hint_low, hint_high)
    ("Smile",       "Lip corner puller", "AU12", "flat mouth",     "smiling"),
    ("Cheek raise", "Cheek raiser",      "AU6",  "relaxed cheeks", "genuine smile"),
    ("Brow furrow", "Brow lowerer",      "AU4",  "relaxed brow",   "frowning / focused"),
    ("Brow raise",  "Inner brow raiser", "AU1",  "relaxed brow",   "surprised / worried"),
    ("Mouth open",  "Lips part",         "AU25", "closed mouth",   "talking / surprised"),
]

STATE = {
    "mouse": (0, 0),
    "mouse_small_x": 0, "mouse_small_y": 0,
    "panel": "kernel",
    "kernel_idx": 0, "se_idx": 0, "op_idx": 0,
    "mode": "LAB",
    "show_liveness": True,
    "frame_count": 0,
    "otsu_cache": None, "hough_cache": None,
    "math_cache": None, "math_sig": None,
    "focus_log": [],
    "start_time": None,
    "blink_times": collections.deque(maxlen=600),
    "mood_kf": None,
    "prev_gray": None, "prev_pts": None,
    "face_box": None,
}


# ============================================================================
# DRAW UTILITIES
# ============================================================================
def blank_canvas(w, h, bg=C_BG):
    c = np.zeros((h, w, 3), dtype=np.uint8); c[:] = bg; return c


def draw_badge(canvas, x, y, num, radius=22, active=True):
    bg_col = C_AMBER if active else (70, 60, 55)
    cv2.circle(canvas, (x, y), radius, bg_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), radius, C_TEXT, 2, cv2.LINE_AA)
    txt = str(num)
    (tw, th), _ = cv2.getTextSize(txt, F_BODY, 1.0, 2)
    cv2.putText(canvas, txt, (x - tw // 2, y + th // 2),
                F_BODY, 1.0, (20, 20, 20), 2, cv2.LINE_AA)


def draw_centered(canvas, text, cx, cy, color=C_TEXT, scale=0.7,
                  font=F_BODY, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2),
                font, scale, color, thickness, cv2.LINE_AA)


def draw_number_grid(canvas, values, x0, y0, cw, ch,
                     text_color_fn=None, border_color=C_BORDER,
                     bg_fn=None, strike_fn=None, text_scale=0.85):
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            cx0, cy0 = x0 + c * cw, y0 + r * ch
            cx1, cy1 = cx0 + cw, cy0 + ch
            bg = bg_fn(r, c) if bg_fn else C_PANEL
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), bg, -1)
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), border_color, 2)
            if strike_fn and strike_fn(r, c):
                cv2.line(canvas, (cx0 + 5, cy0 + 5), (cx1 - 5, cy1 - 5),
                         C_ZERO, 2, cv2.LINE_AA)
            v = values[r, c]
            if isinstance(v, (float, np.floating)) and v != int(v):
                txt = f"{v:.2f}".rstrip("0").rstrip(".")
            else:
                txt = str(int(v))
            color = text_color_fn(v) if text_color_fn else C_TEXT
            draw_centered(canvas, txt, (cx0 + cx1) // 2, (cy0 + cy1) // 2,
                          color=color, scale=text_scale, thickness=2)


def draw_legend(canvas):
    w, h = canvas.shape[1], canvas.shape[0]
    y0 = h - 56
    cv2.rectangle(canvas, (0, y0), (w, h), (14, 12, 12), -1)
    cv2.line(canvas, (0, y0), (w, y0), C_BORDER, 1)
    x, ymid = 20, y0 + 28

    def chip(color, label):
        nonlocal x
        cv2.rectangle(canvas, (x, ymid - 9), (x + 18, ymid + 9), color, -1)
        x += 26
        cv2.putText(canvas, label, (x, ymid + 7), F_BODY, 0.55,
                    C_TEXT, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, F_BODY, 0.55, 1)
        x += tw + 24

    chip(C_POS, "positive"); chip(C_NEG, "negative"); chip(C_ZERO, "zero")
    cv2.drawMarker(canvas, (x + 9, ymid), C_YELLOW,
                   cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    x += 26
    cv2.putText(canvas, "mouse picks pixel", (x, ymid + 7),
                F_BODY, 0.55, C_TEXT, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("mouse picks pixel", F_BODY, 0.55, 1)
    x += tw + 24
    cv2.putText(canvas,
                "[1-4] panel  [k] kernel  [m] SE  [o] dil/ero  "
                "[l] LAB/DEMO  [q] quit",
                (x, ymid + 7), F_BODY, 0.55, C_AMBER, 1, cv2.LINE_AA)


def draw_section_header(canvas, x0, y0, col_w, num, title, subtitle,
                        active=True):
    draw_badge(canvas, x0 + 34, y0 + 30, num, radius=24, active=active)
    cv2.putText(canvas, title, (x0 + 70, y0 + 40),
                F_TITLE, 0.85, C_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (x0 + 18, y0 + 78),
                F_BODY, 0.55, C_MUTED, 1, cv2.LINE_AA)


# ============================================================================
# PATCH EXTRACTION
# ============================================================================
def extract_patch(gray, x, y, k):
    h, w = gray.shape
    r = k // 2
    patch = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            yy, xx = y + i - r, x + j - r
            if 0 <= yy < h and 0 <= xx < w:
                patch[i, j] = gray[yy, xx]
    return patch


# ============================================================================
# PANEL 1 — KERNEL
# ============================================================================
def convolve_at(gray, x, y, kernel):
    k = kernel.shape[0]
    patch = extract_patch(gray, x, y, k)
    products = patch * kernel
    raw = float(products.sum())
    return patch, products, raw, int(np.clip(abs(raw), 0, 255))


def render_kernel_panel(gray, x, y, kernel_name, cw, ch):
    kernel = KERNELS[kernel_name]
    k = kernel.shape[0]
    patch, products, raw_sum, clipped = convolve_at(gray, x, y, kernel)

    canvas = blank_canvas(cw, ch)
    header_h, legend_h = 100, 56
    body_h = ch - header_h - legend_h
    half_w = cw // 2
    half_h = body_h // 2

    avail_w = half_w - 40
    avail_h = half_h - 100
    cell = min(avail_w // max(k, 1), avail_h // max(k, 1), 90)

    def kcolor(v):
        if v > 0: return C_POS
        if v < 0: return C_NEG
        return C_ZERO

    draw_section_header(canvas, 0, 0, half_w, 1, "PIXEL NEIGHBORHOOD",
                        f"The {k}x{k} pixels around ({x}, {y})")
    gx = (half_w - k * cell) // 2
    gy = 100
    draw_number_grid(canvas, patch, gx, gy, cell, cell,
                     text_color_fn=lambda v: C_POS if v > 128 else C_AMBER,
                     text_scale=1.0)
    cv2.line(canvas, (half_w, 0), (half_w, header_h + body_h), C_BORDER, 1)
    cv2.line(canvas, (0, half_h), (cw, half_h), C_BORDER, 1)

    draw_section_header(canvas, half_w, 0, half_w, 2, "KERNEL",
                        f"The operator '{kernel_name}' applied to it")
    gx = half_w + (half_w - k * cell) // 2
    gy = 100
    draw_number_grid(canvas, kernel, gx, gy, cell, cell,
                     text_color_fn=kcolor, border_color=C_AMBER,
                     text_scale=1.0)

    draw_section_header(canvas, 0, half_h, half_w, 3, "PRODUCTS",
                        "Multiply each cell pair")
    gx = (half_w - k * cell) // 2
    gy = half_h + 100
    draw_number_grid(canvas, products, gx, gy, cell, cell,
                     text_color_fn=kcolor, text_scale=1.0)

    draw_section_header(canvas, half_w, half_h, half_w, 4, "RESULT",
                        "Sum all products -> output pixel")
    eq_x = half_w + 30
    eq_y = half_h + 130
    cv2.putText(canvas, "G(x,y) = Sum Sum I*K",
                (eq_x, eq_y), F_TITLE, 0.85, C_AMBER, 2, cv2.LINE_AA)
    eq_y += 44
    terms = []
    for i in range(k):
        for j in range(k):
            c = kernel[i, j]
            if c == 0: continue
            s = "+" if c > 0 else "-"
            cc = abs(int(c)) if c == int(c) else abs(c)
            terms.append(f"{s} {int(patch[i,j])}.{cc}")
    expr = " ".join(terms)
    if expr.startswith("+ "): expr = expr[2:]
    lines, cur = [], ""
    for tok in expr.split(" "):
        if len(cur) + len(tok) + 1 > 32:
            lines.append(cur); cur = tok
        else:
            cur = (cur + " " + tok) if cur else tok
    if cur: lines.append(cur)
    for line in lines[:3]:
        cv2.putText(canvas, line, (eq_x, eq_y), F_BODY, 0.6,
                    C_TEXT, 1, cv2.LINE_AA)
        eq_y += 30
    eq_y += 12
    cv2.putText(canvas, f"sum = {int(raw_sum)}",
                (eq_x, eq_y), F_TITLE, 1.4, C_POS, 3, cv2.LINE_AA)
    eq_y += 52
    cv2.putText(canvas, f"clip(|sum|) = {clipped}",
                (eq_x, eq_y), F_BODY, 0.6, C_MUTED, 1, cv2.LINE_AA)
    eq_y += 32
    cv2.putText(canvas, "-> output pixel",
                (eq_x, eq_y), F_BODY, 0.6, C_GREEN, 1, cv2.LINE_AA)

    draw_legend(canvas)
    return canvas


# ============================================================================
# PANEL 2 — MORPHOLOGY
# ============================================================================
def morph_at(gray, x, y, se, op):
    k = se.shape[0]
    patch = extract_patch(gray, x, y, k)
    mask = se > 0
    active = patch[mask]
    result = (float(active.max()) if op == "dilate"
              else float(active.min())) if active.size else 0.0
    return patch, active, result


def render_morph_panel(gray, x, y, se_name, op, cw, ch):
    se = SE_KERNELS[se_name]
    k = se.shape[0]
    patch, active, result = morph_at(gray, x, y, se, op)

    canvas = blank_canvas(cw, ch)
    header_h, legend_h = 100, 56
    body_h = ch - header_h - legend_h
    half_w = cw // 2
    half_h = body_h // 2

    avail_w = half_w - 40
    avail_h = half_h - 100
    cell = min(avail_w // max(k, 1), avail_h // max(k, 1), 80)

    draw_section_header(canvas, 0, 0, half_w, 1, "PIXEL NEIGHBORHOOD",
                        f"The {k}x{k} pixels around ({x}, {y})")
    gx = (half_w - k * cell) // 2
    gy = 100
    draw_number_grid(canvas, patch, gx, gy, cell, cell,
                     text_color_fn=lambda v: C_TEXT)
    cv2.line(canvas, (half_w, 0), (half_w, header_h + body_h), C_BORDER, 1)
    cv2.line(canvas, (0, half_h), (cw, half_h), C_BORDER, 1)

    draw_section_header(canvas, half_w, 0, half_w, 2, "STRUCTURING ELEMENT",
                        f"Shape '{se_name}'", active=True)
    gx = half_w + (half_w - k * cell) // 2
    gy = 100
    draw_number_grid(canvas, se.astype(np.float32), gx, gy, cell, cell,
                     text_color_fn=lambda v: C_POS if v > 0 else C_ZERO,
                     border_color=C_AMBER)

    draw_section_header(canvas, 0, half_h, half_w, 3, "ACTIVE VALUES",
                        "Only cells under the SE footprint")
    gx = (half_w - k * cell) // 2
    gy = half_h + 100
    draw_number_grid(canvas, patch, gx, gy, cell, cell,
                     text_color_fn=lambda v: C_POS if v > 0 else C_MUTED,
                     bg_fn=lambda r, c: C_PANEL_HI if se[r, c] > 0 else C_PANEL,
                     strike_fn=lambda r, c: se[r, c] == 0)

    red = "max" if op == "dilate" else "min"
    verb = "dilation" if op == "dilate" else "erosion"
    draw_section_header(canvas, half_w, half_h, half_w, 4, "REDUCTION",
                        f"Reduce with {red} ({verb})")
    eq_x = half_w + 30
    eq_y = half_h + 130
    op_sym = "(f (+) b)" if op == "dilate" else "(f (-) b)"
    cv2.putText(canvas, f"{verb.capitalize()} {op_sym}",
                (eq_x, eq_y), F_TITLE, 0.85, C_AMBER, 2, cv2.LINE_AA)
    eq_y += 44
    vals = [int(v) for v in active[:10]]
    vstr = ", ".join(str(v) for v in vals)
    if active.size > 10:
        vstr += f", ... ({active.size} total)"
    cv2.putText(canvas, f"= {red}{{ {vstr} }}",
                (eq_x, eq_y), F_BODY, 0.6, C_TEXT, 1, cv2.LINE_AA)
    eq_y += 60
    cv2.putText(canvas, f"= {int(result)}",
                (eq_x, eq_y), F_TITLE, 1.4, C_POS, 3, cv2.LINE_AA)
    eq_y += 52
    cv2.putText(canvas, "-> output pixel",
                (eq_x, eq_y), F_BODY, 0.6, C_GREEN, 1, cv2.LINE_AA)

    draw_legend(canvas)
    return canvas


# ============================================================================
# PANEL 3 — OTSU
# ============================================================================
def compute_otsu(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = int(hist.sum())
    if total == 0:
        return 0, np.zeros(256), hist, 0.0
    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)
    lv = np.arange(256, dtype=np.float64)
    mu_cum = np.cumsum(lv * p)
    mu_T = mu_cum[-1]
    denom = np.maximum(omega * (1.0 - omega), 1e-12)
    sigma_B2 = (mu_T * omega - mu_cum) ** 2 / denom
    return int(np.argmax(sigma_B2)), sigma_B2, hist, mu_T


def fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR).copy()
    plt.close(fig)
    return bgr


def render_otsu_panel(gray, cw, ch):
    T_star, sigma_B2, hist, mu_T = compute_otsu(gray)
    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100,
                     facecolor="#121218")
    gs = GridSpec(1, 3, figure=fig, wspace=0.28,
                  left=0.05, right=0.98, top=0.85, bottom=0.15)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1a1a24")
    ax1.bar(np.arange(256), hist, width=1.0, color="#4ea8de")
    ax1.axvline(T_star, color="#ff4757", linestyle="--", linewidth=2)
    ax1.set_title(f"Histogram  mu_T = {mu_T:.1f}", color="#e8e8f0", fontsize=14)
    ax1.tick_params(colors="#c8c8d8", labelsize=11)
    for s in ax1.spines.values(): s.set_color("#5a5a6e")

    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#1a1a24")
    ax2.plot(sigma_B2, color="#7dffa0", linewidth=3)
    ax2.axvline(T_star, color="#ff4757", linestyle="--", linewidth=2)
    ax2.plot(T_star, sigma_B2[T_star], "o", color="#ff4757", markersize=12)
    ax2.set_title("sigma_B^2(T)", color="#e8e8f0", fontsize=14)
    ax2.tick_params(colors="#c8c8d8", labelsize=11)
    for s in ax2.spines.values(): s.set_color("#5a5a6e")

    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#1a1a24")
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values(): s.set_color("#5a5a6e")
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    o0 = p[: T_star + 1].sum(); o1 = 1.0 - o0
    lv = np.arange(256, dtype=np.float64)
    m0 = (lv[: T_star + 1] * p[: T_star + 1]).sum() / max(o0, 1e-12)
    m1 = (lv[T_star + 1:] * p[T_star + 1:]).sum() / max(o1, 1e-12)
    sb2 = o0 * o1 * (m0 - m1) ** 2
    ax3.text(0.03, 0.90, "Otsu's optimum", color="#ffd166",
             fontsize=16, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.03, 0.74, f"T* = {T_star}", color="#e8e8f0",
             fontsize=14, family="monospace", transform=ax3.transAxes)
    ax3.text(0.03, 0.56, f"w0={o0:.3f}  m0={m0:.2f}", color="#7dffa0",
             fontsize=12, family="monospace", transform=ax3.transAxes)
    ax3.text(0.03, 0.44, f"w1={o1:.3f}  m1={m1:.2f}", color="#7dffa0",
             fontsize=12, family="monospace", transform=ax3.transAxes)
    ax3.text(0.03, 0.08, f"sigma_B^2 = {sb2:.1f}", color="#ff4757",
             fontsize=20, fontweight="bold", transform=ax3.transAxes)
    return fig_to_bgr(fig)


# ============================================================================
# PANEL 4 — HOUGH
# ============================================================================
def hough_lines(edges, n=3, min_votes=20):
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag, diag + 1, 1.0)
    thetas = np.arange(0, np.pi, np.pi / 180.0)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    ys, xs = np.nonzero(edges)
    if len(xs) == 0: return acc, rhos, thetas, []
    if len(xs) > 1500:
        idx = np.random.choice(len(xs), 1500, replace=False)
        xs, ys = xs[idx], ys[idx]
    ct, st = np.cos(thetas), np.sin(thetas)
    for x, y in zip(xs, ys):
        r = x * ct + y * st
        ri = np.round(r + diag).astype(int)
        ok = (ri >= 0) & (ri < len(rhos))
        acc[ri[ok], np.arange(len(thetas))[ok]] += 1
    aw = acc.copy(); peaks = []
    for _ in range(n):
        if aw.max() < min_votes: break
        ri, ti = np.unravel_index(np.argmax(aw), aw.shape)
        peaks.append((float(rhos[ri]), float(thetas[ti]), int(aw[ri, ti])))
        aw[max(0, ri - 20):ri + 21, max(0, ti - 10):ti + 11] = 0
    return acc, rhos, thetas, peaks


def render_hough_panel(edges, cw, ch):
    acc, rhos, thetas, peaks = hough_lines(edges)
    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100,
                     facecolor="#121218")
    gs = GridSpec(1, 2, figure=fig, wspace=0.22,
                  left=0.05, right=0.98, top=0.85, bottom=0.15)
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1a1a24")
    ax1.imshow(edges, cmap="gray")
    ax1.set_title("Edge map + detected lines", color="#e8e8f0", fontsize=14)
    ax1.set_xticks([]); ax1.set_yticks([])
    h, w = edges.shape
    for (rho, theta, _) in peaks:
        c, s = np.cos(theta), np.sin(theta)
        ax1.plot([0, w - 1], [rho / max(s, 1e-9),
                              (rho - (w - 1) * c) / max(s, 1e-9)],
                 "-", color="#ff4757", linewidth=2)
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#1a1a24")
    td = np.degrees(thetas)
    ax2.imshow(acc, cmap="inferno", aspect="auto",
               extent=[td[0], td[-1], rhos[0], rhos[-1]], origin="lower")
    ax2.set_xlabel("theta (deg)", color="#c8c8d8", fontsize=11)
    ax2.set_ylabel("rho (px)", color="#c8c8d8", fontsize=11)
    ax2.set_title("Accumulator A(rho, theta)", color="#e8e8f0", fontsize=14)
    ax2.tick_params(colors="#c8c8d8", labelsize=11)
    for s in ax2.spines.values(): s.set_color("#5a5a6e")
    for i, (rho, theta, v) in enumerate(peaks):
        ax2.plot(np.degrees(theta), rho, "o", markerfacecolor="none",
                 markeredgecolor="#7dffa0", markeredgewidth=2.5,
                 markersize=14)
        ax2.text(np.degrees(theta) + 4, rho, f"  #{i+1}: {v}",
                 color="#7dffa0", fontsize=10, fontweight="bold")
    return fig_to_bgr(fig)


# ============================================================================
# FACE DETECT
# ============================================================================
def load_face_detector(script_dir):
    proto = os.path.join(script_dir, "deploy.prototxt")
    model = os.path.join(script_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        try:
            net = cv2.dnn.readNetFromCaffe(proto, model)
            print(f"[focusguard] DNN face detector loaded")
            return "dnn", net
        except cv2.error as e:
            print(f"[focusguard] DNN failed: {e}")
    haar = os.path.join(cv2.data.haarcascades,
                        "haarcascade_frontalface_default.xml")
    if os.path.exists(haar):
        print("[focusguard] Using Haar cascade fallback")
        return "haar", cv2.CascadeClassifier(haar)
    return "none", None


def detect_face_dnn(net, frame_bgr, conf=0.7):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()
    best, best_area = None, 0
    for i in range(det.shape[2]):
        c = float(det[0, 0, i, 2])
        if c < conf: continue
        b = det[0, 0, i, 3:7] * np.array([w, h, w, h])
        area = (b[2] - b[0]) * (b[3] - b[1])
        if area < 0.02 * w * h: continue
        if area > best_area:
            best = b.astype(int)
            best_area = area
    return best


def detect_face_haar(haar, gray_small):
    boxes = haar.detectMultiScale(gray_small, scaleFactor=1.15,
                                  minNeighbors=6, minSize=(80, 80))
    if len(boxes) == 0: return None
    h, w = gray_small.shape[:2]
    cx, cy = w / 2, h / 2
    best, best_score = None, -1
    for (x, y, bw, bh) in boxes:
        bx, by = x + bw / 2, y + bh / 2
        d = np.hypot(bx - cx, by - cy)
        score = (bw * bh) - d * 50
        if score > best_score:
            best = np.array([x, y, x + bw, y + bh], dtype=int)
            best_score = score
    return best


# ============================================================================
# MOOD / LIVENESS
# ============================================================================
def mouth_au(gray, roi):
    if roi.size < 100:
        return 0.0, 0.0, np.zeros_like(gray)
    T, _, _, _ = compute_otsu(roi)
    _, mask = cv2.threshold(roi, T, 255, cv2.THRESH_BINARY_INV)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0, 0.0, mask
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    roi_area = float(roi.shape[0] * roi.shape[1])
    if area < 0.03 * roi_area or area > 0.55 * roi_area:
        return 0.0, 0.0, mask
    if len(c) < 5: return 0.0, 0.0, mask
    pts = c.reshape(-1, 2)
    try:
        a = float(np.polyfit(pts[:, 0].astype(np.float64),
                             pts[:, 1].astype(np.float64), 2)[0])
    except Exception:
        a = 0.0
    _, _, bw, bh = cv2.boundingRect(c)
    if bh > 0.5 * bw:
        return 0.0, 0.0, mask
    au12 = float(np.clip(-a * 200.0, 0.0, 1.0))
    au25 = float(np.clip(bh / max(bw, 1) / 0.6, 0.0, 1.0))
    return au12, au25, mask


def eye_au(gray, roi):
    if roi.size < 100: return 0.0, np.zeros_like(gray)
    mask = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 5)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    dr = float((mask > 0).sum()) / max(mask.size, 1)
    return float(np.clip(1.0 - dr / 0.25, 0.0, 1.0)), mask


def brow_au(gray, roi):
    if roi.size < 100: return 0.0, 0.0, np.zeros_like(gray)
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                            minLineLength=15, maxLineGap=8)
    if lines is None or len(lines) == 0: return 0.0, 0.0, edges
    sgn = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
           for l in lines]
    m = float(np.mean(sgn))
    return (float(np.clip(m / 20.0, 0.0, 1.0)),
            float(np.clip(-m / 20.0, 0.0, 1.0)), edges)


def init_kalman(n=5):
    kf = cv2.KalmanFilter(n, n)
    kf.transitionMatrix = np.eye(n, dtype=np.float32)
    kf.measurementMatrix = np.eye(n, dtype=np.float32)
    kf.processNoiseCov = np.eye(n, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(n, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(n, dtype=np.float32)
    kf.statePost = np.zeros((n, 1), dtype=np.float32)
    return kf


def classify_mood(au):
    a12, a6, a4, a1, a25 = au
    if a12 > 0.40 and a6 > 0.15:  return "happy"
    if a4 > 0.40 and a1 < 0.20:   return "focused"
    if a1 > 0.35 and a25 > 0.25:  return "surprised"
    if a25 > 0.40:                return "talking"
    return "neutral"


LK = dict(winSize=(15, 15), maxLevel=2,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def liveness_flow(gray, box):
    if box is None:
        STATE["prev_gray"] = gray; STATE["prev_pts"] = None
        return 0.0
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = gray.shape
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 - x1 < 20 or y2 - y1 < 20: return 0.0
    m = np.zeros_like(gray); m[y1:y2, x1:x2] = 255
    pg, pp = STATE["prev_gray"], STATE["prev_pts"]
    if pg is None or pp is None or len(pp) < 5:
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=30,
                                      qualityLevel=0.01, minDistance=10,
                                      mask=m)
        STATE["prev_gray"] = gray; STATE["prev_pts"] = pts
        return 0.0
    np_, st, _ = cv2.calcOpticalFlowPyrLK(pg, gray, pp, None, **LK)
    STATE["prev_gray"] = gray
    if np_ is None or st is None:
        STATE["prev_pts"] = None; return 0.0
    good = st.ravel() == 1
    if good.sum() < 3:
        STATE["prev_pts"] = np_; return 0.0
    f = float(np.linalg.norm(np_[good] - pp[good], axis=2).mean())
    STATE["prev_pts"] = np_
    return f


def analyze_face(gray_small):
    box = STATE["face_box"]
    res = {"face_box": box, "au": np.zeros(5, dtype=np.float32),
           "mood": "neutral", "focus": 0.0, "motion": 0.0}
    if box is None: return res
    h, w = gray_small.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    fh, fw = y2 - y1, x2 - x1
    if fh < 60 or fw < 60: return res

    brow_y0, brow_y1 = y1 + int(fh * 0.15), y1 + int(fh * 0.35)
    eye_y0, eye_y1 = y1 + int(fh * 0.35), y1 + int(fh * 0.55)
    mouth_y0, mouth_y1 = y1 + int(fh * 0.70), y1 + int(fh * 0.95)
    mouth_x0, mouth_x1 = x1 + int(fw * 0.25), x1 + int(fw * 0.75)

    brow_roi = gray_small[brow_y0:brow_y1, x1:x2]
    eye_roi = gray_small[eye_y0:eye_y1, x1:x2]
    mouth_roi = gray_small[mouth_y0:mouth_y1, mouth_x0:mouth_x1]

    au12, au25, _ = mouth_au(gray_small, mouth_roi)
    au6, _ = eye_au(gray_small, eye_roi)
    au1, au4, _ = brow_au(gray_small, brow_roi)

    raw = np.array([au12, au6, au4, au1, au25], dtype=np.float32)
    if STATE["mood_kf"] is None:
        STATE["mood_kf"] = init_kalman(5)
    STATE["mood_kf"].correct(raw.reshape(-1, 1).astype(np.float32))
    smooth = np.clip(STATE["mood_kf"].predict().ravel(), 0.0, 1.0)

    motion = liveness_flow(gray_small, box) if STATE["show_liveness"] else 0.0
    now = time.time()
    if float(raw[1]) > 0.85:
        last = STATE["blink_times"][-1] if STATE["blink_times"] else 0
        if now - last > 0.20:
            STATE["blink_times"].append(now)
    br = sum(1 for t in STATE["blink_times"] if now - t < 60.0)
    focus = float(np.clip(100.0 - min(1.5 * br, 40.0) - 60.0 * motion,
                          0.0, 100.0))
    res.update({"au": smooth, "mood": classify_mood(smooth),
                "focus": focus, "motion": motion})
    return res


# ============================================================================
# OVERLAY — non-expert readable
# ============================================================================
MOOD_COLORS = {"happy": C_POS, "focused": C_BLUE, "surprised": C_AMBER,
               "talking": C_PURPLE, "neutral": C_MUTED}

MOOD_MEANING = {
    "happy":     "smiling with cheeks raised",
    "focused":   "brow lowered, concentrated look",
    "surprised": "brows raised, mouth open",
    "talking":   "mouth moving / open",
    "neutral":   "no strong expression detected",
}


def _au_state(v):
    """Return (label, color) for a raw AU activation value."""
    if v > 0.60:  return "HIGH", C_POS
    if v > 0.30:  return "MED",  C_AMBER
    return "LOW", C_MUTED


def draw_overlay(canvas, data, frame_rect, mouse_xy):
    H, W = canvas.shape[:2]
    fx, fy, fw, fh = frame_rect

    # ---------- Header strip ----------
    hh = 64
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (W, hh), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.80, canvas, 0.20, 0, canvas)
    cv2.putText(canvas, "FocusGuard", (24, 44), F_TITLE, 1.0,
                C_TEXT, 2, cv2.LINE_AA)

    mood_lbl = f"Mood: {data['mood']}"
    mc = MOOD_COLORS.get(data["mood"], C_TEXT)
    (tw, _), _ = cv2.getTextSize(mood_lbl, F_TITLE, 1.0, 2)
    cv2.putText(canvas, mood_lbl, (W - tw - 24, 44), F_TITLE, 1.0,
                mc, 2, cv2.LINE_AA)

    # ---------- Face box ----------
    if data["face_box"] is not None:
        bx1, by1, bx2, by2 = [int(v) for v in data["face_box"]]
        sx, sy = fw / WORK_W, fh / WORK_H
        cv2.rectangle(canvas,
                      (fx + int(bx1 * sx), fy + int(by1 * sy)),
                      (fx + int(bx2 * sx), fy + int(by2 * sy)),
                      C_GREEN, 3, cv2.LINE_AA)

    # ---------- Cursor crosshair ----------
    mx, my = mouse_xy
    if 0 <= mx < W and 0 <= my < H:
        cv2.drawMarker(canvas, (mx, my), C_YELLOW,
                       cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)

    # ---------- LEFT PANEL: what the detector is reading ----------
    panel_x = 20
    panel_y = hh + 20
    panel_w = 380
    panel_h = 300

    ov = canvas.copy()
    cv2.rectangle(ov, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (12, 12, 18), -1)
    cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), C_BORDER, 2,
                  cv2.LINE_AA)

    # Section heading
    cv2.putText(canvas, "FACIAL SIGNALS",
                (panel_x + 16, panel_y + 30), F_TITLE, 0.68,
                C_AMBER, 2, cv2.LINE_AA)
    cv2.putText(canvas, "what the demo is reading on your face",
                (panel_x + 16, panel_y + 54), F_BODY, 0.45,
                C_MUTED, 1, cv2.LINE_AA)
    cv2.line(canvas, (panel_x + 16, panel_y + 62),
             (panel_x + panel_w - 16, panel_y + 62), C_BORDER, 1)

    # AU rows
    bar_x = panel_x + 150
    bar_w = 150
    bar_h = 14
    row_y = panel_y + 80
    row_gap = 42

    for i, (short, plain, tech, hint_lo, hint_hi) in enumerate(AU_INFO):
        v = float(data["au"][i])
        state, scol = _au_state(v)
        y = row_y + i * row_gap

        # Colored dot indicator
        cv2.circle(canvas, (panel_x + 24, y + 10), 6, scol, -1,
                   cv2.LINE_AA)

        # Friendly name
        cv2.putText(canvas, short, (panel_x + 40, y + 16),
                    F_TITLE, 0.62, C_TEXT, 2, cv2.LINE_AA)

        # Bar
        cv2.rectangle(canvas, (bar_x, y + 2),
                      (bar_x + bar_w, y + 2 + bar_h), (40, 40, 52), -1)
        fill = int(bar_w * np.clip(v, 0, 1))
        cv2.rectangle(canvas, (bar_x, y + 2),
                      (bar_x + fill, y + 2 + bar_h), scol, -1)

        # State label + value
        cv2.putText(canvas, state, (bar_x + bar_w + 10, y + 14),
                    F_BODY, 0.52, scol, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{v:.2f}",
                    (bar_x + bar_w + 10, y + 30),
                    F_BODY, 0.42, C_MUTED, 1, cv2.LINE_AA)

        # Technical annotation (subtle, under the row)
        cv2.putText(canvas, f"{plain}  ({tech})",
                    (panel_x + 40, y + 32),
                    F_BODY, 0.40, C_MUTED, 1, cv2.LINE_AA)

    # ---------- RIGHT PANEL: focus score + mood explanation ----------
    card_w = 280
    card_h = 190
    cx = W - card_w - 20
    cy = hh + 20

    ov = canvas.copy()
    cv2.rectangle(ov, (cx, cy), (cx + card_w, cy + card_h),
                  (12, 12, 18), -1)
    cv2.addWeighted(ov, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h),
                  C_BORDER, 2, cv2.LINE_AA)

    cv2.putText(canvas, "FOCUS SCORE", (cx + 16, cy + 30),
                F_TITLE, 0.62, C_AMBER, 2, cv2.LINE_AA)
    cv2.putText(canvas, "(how engaged you look)", (cx + 16, cy + 50),
                F_BODY, 0.42, C_MUTED, 1, cv2.LINE_AA)

    fs = int(data["focus"])
    fcol = C_GREEN if fs > 65 else (C_AMBER if fs > 35 else C_RED)
    # Big number — right aligned
    big = str(fs)
    (bw, bh), _ = cv2.getTextSize(big, F_TITLE, 2.2, 3)
    cv2.putText(canvas, big, (cx + card_w - bw - 16, cy + 120),
                F_TITLE, 2.2, fcol, 3, cv2.LINE_AA)

    # Focus bar
    bar_x2 = cx + 16
    bar_y2 = cy + 138
    bar_w2 = card_w - 32
    cv2.rectangle(canvas, (bar_x2, bar_y2),
                  (bar_x2 + bar_w2, bar_y2 + 12), (40, 40, 52), -1)
    cv2.rectangle(canvas, (bar_x2, bar_y2),
                  (bar_x2 + int(bar_w2 * fs / 100), bar_y2 + 12), fcol, -1)
    cv2.putText(canvas, "0", (bar_x2, bar_y2 + 30),
                F_BODY, 0.35, C_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "100", (bar_x2 + bar_w2 - 24, bar_y2 + 30),
                F_BODY, 0.35, C_MUTED, 1, cv2.LINE_AA)

    # ---------- Bottom strip: mood explanation ----------
    strip_h = 56
    sy = H - strip_h
    ov = canvas.copy()
    cv2.rectangle(ov, (0, sy), (W, H), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.80, canvas, 0.20, 0, canvas)
    cv2.line(canvas, (0, sy), (W, sy), C_BORDER, 1)

    why = MOOD_MEANING.get(data["mood"], "")
    cv2.putText(canvas, "Why this mood?",
                (20, sy + 24), F_TITLE, 0.55, C_AMBER, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"'{data['mood']}' = {why}",
                (20, sy + 46), F_BODY, 0.55, C_TEXT, 1, cv2.LINE_AA)

    # Liveness indicator (bottom-right)
    if STATE["show_liveness"]:
        live = "LIVE PERSON" if data["motion"] > 0.15 else "STILL (spoof?)"
        lc = C_GREEN if data["motion"] > 0.15 else C_RED
        (lw, _), _ = cv2.getTextSize(live, F_TITLE, 0.62, 2)
        cv2.putText(canvas, live, (W - lw - 20, sy + 30),
                    F_TITLE, 0.62, lc, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"motion={data['motion']:.2f} px/frame",
                    (W - 220, sy + 48), F_BODY, 0.45, C_MUTED, 1,
                    cv2.LINE_AA)


# ============================================================================
# WINDOW SETUP
# ============================================================================
def get_screen():
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def compute_layout(sw, sh):
    M, G, T, B = 20, 20, 20, 60
    aw, ah = sw - 2 * M - G, sh - T - B
    vert = sw < 1200
    if vert:
        mw = aw + G; a_w = mw; mh = int(ah * 0.45); a_h = ah - mh - G
        return {"vertical": True, "margin": M, "gutter": G, "top": T,
                "main": (M, T, mw, mh), "math": (M, T + mh + G, a_w, a_h)}
    mw = int(aw * 0.40); a_w = aw - mw; mh = ah; a_h = ah
    return {"vertical": False, "margin": M, "gutter": G, "top": T,
            "main": (M, T, mw, mh),
            "math": (M + mw + G, T, a_w, a_h)}


def fit_frame(frame, rw, rh):
    fh, fw = frame.shape[:2]
    s = min(rw / fw, rh / fh)
    nw, nh = int(fw * s), int(fh * s)
    return (cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
            (rw - nw) // 2, (rh - nh) // 2, nw, nh)


# ============================================================================
# MATH PANEL DISPATCH
# ============================================================================
def math_signature():
    return (STATE["panel"], STATE["kernel_idx"], STATE["se_idx"],
            STATE["op_idx"], STATE["mouse_small_x"], STATE["mouse_small_y"])


def get_math_panel(gray_small, fc, cw, ch):
    sig = math_signature()
    if STATE["math_sig"] == sig and STATE["math_cache"] is not None:
        return STATE["math_cache"]
    p = STATE["panel"]
    x, y = STATE["mouse_small_x"], STATE["mouse_small_y"]
    if p == "kernel":
        c = render_kernel_panel(gray_small, x, y,
                                KERNEL_ORDER[STATE["kernel_idx"]], cw, ch)
    elif p == "morph":
        c = render_morph_panel(gray_small, x, y,
                               SE_ORDER[STATE["se_idx"]],
                               MORPH_OPS[STATE["op_idx"]], cw, ch)
    elif p == "otsu":
        if STATE["otsu_cache"] is None or fc % MATH_REDRAW_EVERY == 0:
            STATE["otsu_cache"] = render_otsu_panel(gray_small, cw, ch)
        c = STATE["otsu_cache"]
    elif p == "hough":
        if STATE["hough_cache"] is None or fc % MATH_REDRAW_EVERY == 0:
            edges = cv2.Canny(gray_small, 50, 150)
            STATE["hough_cache"] = render_hough_panel(edges, cw, ch)
        c = STATE["hough_cache"]
    else:
        c = blank_canvas(cw, ch)
    STATE["math_cache"] = c
    STATE["math_sig"] = sig
    return c


# ============================================================================
# INTERACTION
# ============================================================================
def on_mouse(event, x, y, flags, _):
    if event == cv2.EVENT_MOUSEMOVE:
        STATE["mouse"] = (x, y)


def handle_key(k):
    if k in (255, -1): return True
    if k == ord("q"): return False
    if k == ord("1"): STATE["panel"] = "kernel"
    elif k == ord("2"): STATE["panel"] = "morph"
    elif k == ord("3"): STATE["panel"] = "otsu"
    elif k == ord("4"): STATE["panel"] = "hough"
    elif k == ord("k"):
        STATE["kernel_idx"] = (STATE["kernel_idx"] + 1) % len(KERNEL_ORDER)
    elif k == ord("m"):
        STATE["se_idx"] = (STATE["se_idx"] + 1) % len(SE_ORDER)
    elif k == ord("o"):
        STATE["op_idx"] = (STATE["op_idx"] + 1) % len(MORPH_OPS)
    elif k == ord("l"):
        STATE["mode"] = "DEMO" if STATE["mode"] == "LAB" else "LAB"
    elif k == ord("s"):
        STATE["show_liveness"] = not STATE["show_liveness"]
    STATE["math_sig"] = None
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    kind, detector = load_face_detector(here)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[focusguard] cannot open camera {CAM_INDEX}")
        return

    sw, sh = get_screen()
    L = compute_layout(sw, sh)
    mx_, my_, mw_, mh_ = L["main"]
    ax_, ay_, aw_, ah_ = L["math"]
    print(f"[focusguard] Screen: {sw}x{sh} "
          f"({'vertical' if L['vertical'] else 'horizontal'})")
    print(f"[focusguard] FocusGuard: {mw_}x{mh_} at ({mx_}, {my_})")
    print(f"[focusguard] Math Panel: {aw_}x{ah_} at ({ax_}, {ay_})")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MATH, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, mw_, mh_)
    cv2.resizeWindow(WINDOW_MATH, aw_, ah_)
    cv2.moveWindow(WINDOW_MAIN, mx_, my_)
    cv2.moveWindow(WINDOW_MATH, ax_, ay_)
    cv2.waitKey(1); cv2.waitKey(1)
    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)
    if shutil.which("wmctrl"):
        for w in (WINDOW_MAIN, WINDOW_MATH):
            try:
                subprocess.Popen(["wmctrl", "-r", w, "-b", "add,above"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except Exception: pass

    STATE["start_time"] = time.time()
    print("=" * 64)
    print(" FocusGuard running.  1/2/3/4 panel | k kernel | m SE")
    print(" o dil/ero | l LAB/DEMO | s liveness | q quit")
    print("=" * 64)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            small = cv2.resize(frame, (WORK_W, WORK_H),
                               interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            STATE["frame_count"] += 1
            fc = STATE["frame_count"]

            mxd, myd = STATE["mouse"]
            STATE["mouse_small_x"] = int(np.clip(
                mxd * WORK_W / max(mw_, 1), 0, WORK_W - 1))
            STATE["mouse_small_y"] = int(np.clip(
                myd * WORK_H / max(mh_, 1), 0, WORK_H - 1))

            if fc % DETECT_EVERY == 0:
                if kind == "dnn":
                    STATE["face_box"] = detect_face_dnn(detector, small)
                elif kind == "haar":
                    STATE["face_box"] = detect_face_haar(detector, gray_small)
                else:
                    STATE["face_box"] = None

            mood = analyze_face(gray_small)

            # FocusGuard canvas — draw the frame, then overlay on top
            canvas = blank_canvas(mw_, mh_)
            fitted, xo, yo, fw_, fh_ = fit_frame(frame, mw_, mh_ - 64)
            canvas[64 + yo:64 + yo + fh_, xo:xo + fw_] = fitted
            fr = (xo, 64 + yo, fw_, fh_)

            # face_box stays in SMALL coords — draw_overlay scales it
            # exactly once.
            draw_overlay(canvas, mood, fr, STATE["mouse"])

            math = get_math_panel(gray_small, fc, aw_, ah_)

            cv2.imshow(WINDOW_MAIN, canvas)
            cv2.imshow(WINDOW_MATH, math)

            if STATE["start_time"] is not None:
                t = time.time() - STATE["start_time"]
                STATE["focus_log"].append((t, mood["focus"]))

            k = cv2.waitKey(1) & 0xFF
            if not handle_key(k): break
            if cv2.getWindowProperty(WINDOW_MAIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(WINDOW_MATH, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("[focusguard] Interrupted.")
    finally:
        cap.release(); cv2.destroyAllWindows()
        print("[focusguard] Released.")


# ============================================================================
# SESSION REPORT
# ============================================================================
def session_report(path="session_report.png"):
    log = STATE["focus_log"]
    if not log:
        print("[focusguard] no session data"); return None
    t = [a for a, _ in log]; s = [b for _, b in log]
    fig = plt.figure(figsize=(11, 3.5), dpi=110, facecolor="#121218")
    ax = fig.add_subplot(1, 1, 1); ax.set_facecolor("#1a1a24")
    ax.plot(t, s, color="#7dffa0", linewidth=2)
    ax.fill_between(t, s, alpha=0.15, color="#7dffa0")
    ax.set_xlabel("session time (s)", color="#c8c8d8")
    ax.set_ylabel("focus score", color="#c8c8d8")
    ax.set_title(f"avg focus = {float(np.mean(s)):.1f}",
                 color="#e8e8f0", fontsize=12)
    ax.tick_params(colors="#c8c8d8")
    ax.grid(True, alpha=0.15, color="#5a5a6e")
    for sp in ax.spines.values(): sp.set_color("#5a5a6e")
    ax.set_ylim(0, 100); fig.tight_layout()
    fig.savefig(path, facecolor="#121218"); plt.close(fig)
    print(f"[focusguard] report -> {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        session_report()