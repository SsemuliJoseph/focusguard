#!/usr/bin/env python3
"""
FocusGuard — Live Classical Computer Vision Demo
================================================

A real-time webcam demo that makes classical CV mathematics visible at
the pixel level. Two windows:

  * FocusGuard  — the mood/focus detector (the product)
  * Math Panel  — the exact arithmetic the detector is running (the glass box)

The pedagogical point: every equation in the lecture slides is shown as
actual arithmetic on actual pixels, at 30 fps.

Usage
-----
    python focusguard.py

Controls
--------
    1 / 2 / 3 / 4   Switch math panel (Kernel / Morphology / Otsu / Hough)
    k               Cycle kernel
    m               Cycle structuring element
    o               Toggle dilate <-> erode
    l               Toggle LAB mode <-> DEMO mode
    s               Toggle liveness detection
    q               Quit (runs session report)

Requires: opencv-contrib-python, numpy, matplotlib
"""

# ============================================================================
# SECTION 1 — IMPORTS
# ============================================================================
# Critical: matplotlib MUST be set to Agg before pyplot is imported, otherwise
# VS Code's inline renderer fights cv2.imshow for the event loop and the
# math panels go blank.
# ----------------------------------------------------------------------------
import os
import sys
import time
import collections

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")           # MUST be before `import matplotlib.pyplot`
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# SECTION 2 — CONSTANTS, STATE, KERNELS, STRUCTURING ELEMENTS
# ============================================================================
# All the tunable bits live here so the demo can be adjusted without
# hunting through the functions.
# ----------------------------------------------------------------------------

WINDOW_MAIN = "FocusGuard"
WINDOW_MATH = "Math Panel"
MATH_W, MATH_H = 1400, 420        # math panel canvas size (wide, short)
CAM_INDEX = 0                     # /dev/video0
WORK_W, WORK_H = 480, 360         # all CV work happens at this size
DETECT_EVERY = 3                  # run the DNN face detector every N frames
MATH_REDRAW_EVERY = 6             # redraw matplotlib panels every N frames

# Colors (BGR — OpenCV convention, NOT RGB)
C_BG        = (18, 18, 24)        # dark slate
C_PANEL     = (32, 32, 42)        # panel fill
C_BORDER    = (80, 80, 100)       # panel border
C_TITLE     = (200, 200, 220)     # title text
C_TEXT      = (230, 230, 240)     # body text
C_ACCENT    = (255, 200, 80)      # sky-blue accent (BGR: high B)
C_POS       = (80, 220, 80)       # green — positive
C_NEG       = (80, 80, 240)       # red — negative
C_ZERO      = (140, 140, 140)     # gray — zero
C_GREEN     = (0, 200, 0)
C_RED       = (0, 0, 240)
C_YELLOW    = (0, 220, 220)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Kernel library (Section 5 uses these) ----------------------------------
# Each kernel is a small matrix that we slide over the image. The kernel is
# applied via elementwise multiply-and-sum (convolution / correlation).
KERNELS = {
    "sobel_x":   np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32),
    "sobel_y":   np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32),
    "prewitt_x": np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=np.float32),
    "prewitt_y": np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32),
    "roberts_x": np.array([[ 1, 0],
                           [ 0,-1]], dtype=np.float32),  # 2x2
    "roberts_y": np.array([[ 0, 1],
                           [-1, 0]], dtype=np.float32),
    "laplacian": np.array([[ 0,  1,  0],
                           [ 1, -4,  1],
                           [ 0,  1,  0]], dtype=np.float32),
    "gaussian":  np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32) / 16.0,
    "box_blur":  np.ones((3, 3), dtype=np.float32) / 9.0,
}
KERNEL_ORDER = list(KERNELS.keys())

# --- Structuring elements (Section 6 uses these) ----------------------------
# An SE is a binary footprint. Morphology looks at exactly the pixels under
# the 1s of this footprint and ignores the rest. Shape matters: a disc SE
# is rotationally invariant, a cross SE preserves connectivity, etc.
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

# --- Global UI state (mutated by the mouse and keyboard callbacks) ----------
STATE = {
    "mouse":         (MATH_W // 2, MATH_H // 2),  # (x, y) in FocusGuard window
    "panel":         "kernel",     # kernel | morph | otsu | hough
    "kernel_idx":    0,
    "se_idx":        0,
    "op_idx":        0,
    "mode":          "LAB",        # LAB | DEMO
    "show_liveness": True,
    "frame_count":   0,
    # Cached matplotlib render for Otsu and Hough (blitted into cv2 windows)
    "otsu_cache":    None,
    "hough_cache":   None,
    # Session tracking (used by the report cell)
    "focus_log":     [],           # list of (t_sec, focus_score)
    "start_time":    None,
    "blink_times":   collections.deque(maxlen=600),
    "mood_kf":       None,         # initialized lazily
    "prev_gray":     None,         # for optical flow
    "prev_pts":      None,
    "dnn_net":       None,         # lazily loaded face detector
    "haar":          None,
    "face_box":      None,         # cached face box between detect frames
    "dnn_tried":     False,
}


# ============================================================================
# SECTION 3 — SMALL DRAWING UTILITIES
# ============================================================================
# Everything here is cv2 drawing. We avoid matplotlib for the per-frame
# panels because rendering a figure every frame kills fps. Panels 1 and 2
# are drawn entirely with cv2.line / cv2.rectangle / cv2.putText.
# ----------------------------------------------------------------------------

def blank_canvas(w, h, bg=C_BG):
    """Return a BGR canvas of the given size filled with a dark background."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = bg
    return canvas


def draw_title(canvas, text, x, y, color=C_TITLE, scale=0.55, thickness=1):
    """Draw a small heading with a subtle underline."""
    cv2.putText(canvas, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.line(canvas, (x, y + 4), (x + tw, y + 4), color, 1, cv2.LINE_AA)


def draw_centered_text(canvas, text, cx, cy, color=C_TEXT,
                       scale=0.5, thickness=1):
    """Draw text horizontally centered on cx, vertically on cy."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2),
                FONT, scale, color, thickness, cv2.LINE_AA)


def draw_number_grid(canvas, values, x0, y0, cell_w, cell_h,
                     text_color_fn=None, border_color=C_BORDER):
    """
    Draw a 2D array of numbers as a grid of cells.

    values          : 2D numpy array of numbers (int or float)
    x0, y0          : top-left of the grid on the canvas
    cell_w, cell_h  : pixel size of each cell
    text_color_fn   : optional callable(value) -> BGR color for the text
    """
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h
            # Cell background
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), C_PANEL, -1)
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), border_color, 1)
            # Text
            v = values[r, c]
            if isinstance(v, (float, np.floating)):
                txt = f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 100 else f"{v:.0f}"
            else:
                txt = f"{int(v)}"
            color = text_color_fn(v) if text_color_fn else C_TEXT
            draw_centered_text(canvas, txt, (cx0 + cx1) // 2,
                               (cy0 + cy1) // 2, color=color, scale=0.5)


def draw_equation_card(canvas, x0, y0, x1, y1, header_lines, body_lines):
    """
    Draw a text card with a header and body. Used for the equation panel
    on the right-hand side of panels 1 and 2.

    header_lines : list of (text, scale, color)
    body_lines   : list of (text, scale, color)
    """
    cv2.rectangle(canvas, (x0, y0), (x1, y1), C_PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), C_BORDER, 1)
    y = y0 + 26
    for text, scale, color in header_lines:
        cv2.putText(canvas, text, (x0 + 12, y), FONT, scale, color,
                    1, cv2.LINE_AA)
        y += int(scale * 34) + 6
    y += 4
    for text, scale, color in body_lines:
        cv2.putText(canvas, text, (x0 + 12, y), FONT, scale, color,
                    1, cv2.LINE_AA)
        y += int(scale * 30) + 4


def resize_panel(panel, w, h):
    """Resize a math panel canvas to the target window size."""
    return cv2.resize(panel, (w, h), interpolation=cv2.INTER_AREA)


# ============================================================================
# SECTION 4 — PIXEL NEIGHBORHOOD EXTRACTION (shared by Panels 1 & 2)
# ============================================================================
# This is the single most important helper in the file. It grabs the k x k
# patch centered at (x, y). Out-of-bounds pixels are treated as 0 (zero
# padding), which is what cv2.filter2D does by default at the borders.
# ----------------------------------------------------------------------------

def extract_patch(gray, x, y, k):
    """
    Return the k x k neighborhood centered at (x, y), zero-padded at edges.

    For k = 3 and center (x, y), the patch is:

        [ I(x-1, y-1)  I(x, y-1)  I(x+1, y-1) ]
        [ I(x-1, y  )  I(x, y  )  I(x+1, y  ) ]
        [ I(x-1, y+1)  I(x, y+1)  I(x+1, y+1) ]

    where I(...) is 0 outside the image bounds.
    """
    h, w = gray.shape
    r = k // 2
    patch = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            yy = y + i - r
            xx = x + j - r
            if 0 <= yy < h and 0 <= xx < w:
                patch[i, j] = gray[yy, xx]
    return patch


# ============================================================================
# SECTION 5 — PANEL 1: KERNEL CONVOLUTION ARITHMETIC
# ============================================================================
# The demo: pick a pixel, extract its k x k neighborhood, multiply each
# cell by the corresponding kernel coefficient, sum the products. That sum
# is the output pixel value at the center location.
#
#   G(x, y) = sum_u sum_v  I(x+u, y+v) * K(u, v)
#
# We show four side-by-side regions:
#   1. Neighborhood I(x+u, y+v)
#   2. Kernel K(u, v)
#   3. Elementwise products I * K
#   4. The equation with the actual numbers substituted in
# ----------------------------------------------------------------------------

def convolve_at(gray, x, y, kernel):
    """
    Compute the convolution result at (x, y).

    Returns
    -------
    patch    : k x k neighborhood (float)
    products : k x k elementwise product of patch and kernel
    raw_sum  : the raw sum (may overflow [0, 255])
    clipped  : the value that would be stored in a uint8 image
    """
    k = kernel.shape[0]
    patch = extract_patch(gray, x, y, k)
    products = patch * kernel
    raw_sum = float(products.sum())
    clipped = int(np.clip(abs(raw_sum), 0, 255))  # magnitude, clamped
    return patch, products, raw_sum, clipped


def render_kernel_panel(gray, x, y, kernel_name):
    """Build the full Panel 1 canvas for the given pixel and kernel."""
    kernel = KERNELS[kernel_name]
    k = kernel.shape[0]
    patch, products, raw_sum, clipped = convolve_at(gray, x, y, kernel)

    canvas = blank_canvas(MATH_W, MATH_H)

    # Reserve the four columns. Each column is MATH_W // 4 wide.
    col_w = MATH_W // 4
    pad = 14

    # --- Column 1: neighborhood ---
    x0 = pad
    draw_title(canvas, f"Neighborhood I(x+u, y+v)  @({x},{y})", x0, 26,
               color=C_TITLE)
    cell = min((col_w - 2 * pad) // k, 46)
    grid_x = x0 + (col_w - 2 * pad - k * cell) // 2
    grid_y = 48
    draw_number_grid(
        canvas, patch, grid_x, grid_y, cell, cell,
        text_color_fn=lambda v: C_GREEN if v > 128 else C_ACCENT,
    )

    # --- Column 2: kernel ---
    x0 = pad + col_w
    draw_title(canvas, f"Kernel K[u,v]  ({kernel_name})", x0, 26,
               color=C_TITLE)
    grid_x = x0 + (col_w - 2 * pad - k * cell) // 2
    def kernel_color(v):
        if v > 0:  return C_POS
        if v < 0:  return C_NEG
        return C_ZERO
    draw_number_grid(
        canvas, kernel, grid_x, grid_y, cell, cell,
        text_color_fn=kernel_color,
    )

    # --- Column 3: products ---
    x0 = pad + 2 * col_w
    draw_title(canvas, "Elementwise I * K", x0, 26, color=C_TITLE)
    grid_x = x0 + (col_w - 2 * pad - k * cell) // 2
    def product_color(v):
        if v > 0:  return C_POS
        if v < 0:  return C_NEG
        return C_ZERO
    draw_number_grid(
        canvas, products, grid_x, grid_y, cell, cell,
        text_color_fn=product_color,
    )

    # --- Column 4: equation ---
    x0 = pad + 3 * col_w
    card_x0 = x0
    card_x1 = MATH_W - pad
    card_y0 = 12
    card_y1 = MATH_H - 12

    # Build the equation string with actual numbers substituted in.
    terms = []
    for i in range(k):
        for j in range(k):
            p = int(patch[i, j])
            c = kernel[i, j]
            if c == 0:
                terms.append(f"({p})(0)")
            else:
                sign = "+" if c > 0 else "-"
                terms.append(f"{sign}({p})({abs(int(c)) if c == int(c) else abs(c)})")
    eq = " ".join(terms)
    # Wrap roughly to fit the card
    wrapped = []
    line = ""
    for tok in eq.split(" "):
        if len(line) + len(tok) + 1 > 36:
            wrapped.append(line)
            line = tok
        else:
            line = (line + " " + tok) if line else tok
    if line:
        wrapped.append(line)
    wrapped = wrapped[:6]  # cap so we don't overrun the card

    body = [("= " + wrapped[0], 0.42, C_TEXT)]
    for w_line in wrapped[1:]:
        body.append(("  " + w_line, 0.42, C_TEXT))
    body.append(("", 0.2, C_TEXT))
    body.append((f"sum = {int(raw_sum)}", 0.55, C_ACCENT))
    body.append((f"clip(|sum|) = {clipped}", 0.55, C_GREEN))

    draw_equation_card(
        canvas, card_x0, card_y0, card_x1, card_y1,
        header_lines=[
            ("G(x,y) = \u03A3\u03A3 I \u00B7 K", 0.6, C_TITLE),
        ],
        body_lines=body,
    )

    return canvas


# ============================================================================
# SECTION 6 — PANEL 2: MORPHOLOGY ARITHMETIC
# ============================================================================
# The demo: pick a pixel, extract its k x k neighborhood, and look at ONLY
# the cells where the structuring element is 1. Reduce those values with
# max (dilation) or min (erosion). That reduction is the output pixel.
#
#   Dilation:  (f (+) b)(x) = sup_{h in B}  f(x - h) + b(h)
#   Erosion:   (f (-) b)(x) = inf_{h in B}  f(x + h) - b(h)
#
# For a flat SE (b == 0), these reduce to local max and local min:
#
#   (f (+) b)(x) = max_{h in B} f(x - h)
#   (f (-) b)(x) = min_{h in B} f(x + h)
#
# The panel shows which cells participate in the reduction.
# ----------------------------------------------------------------------------

def morph_at(gray, x, y, se, op):
    """
    Compute the morphological operation at (x, y).

    Returns
    -------
    patch   : k x k neighborhood
    active  : the values under the SE footprint, in scan order
    result  : the max (dilate) or min (erode) of those values
    """
    k = se.shape[0]
    patch = extract_patch(gray, x, y, k)
    mask = se > 0
    active = patch[mask]
    if op == "dilate":
        result = float(active.max()) if active.size else 0.0
    else:
        result = float(active.min()) if active.size else 0.0
    return patch, active, result


def render_morph_panel(gray, x, y, se_name, op):
    """Build the full Panel 2 canvas for the given pixel, SE, and op."""
    se = SE_KERNELS[se_name]
    k = se.shape[0]
    patch, active, result = morph_at(gray, x, y, se, op)

    canvas = blank_canvas(MATH_W, MATH_H)
    col_w = MATH_W // 4
    pad = 14
    cell = min((col_w - 2 * pad) // k, 40)

    # --- Column 1: neighborhood ---
    x0 = pad
    draw_title(canvas, f"Neighborhood  @({x},{y})", x0, 26, color=C_TITLE)
    grid_x = x0 + (col_w - 2 * pad - k * cell) // 2
    grid_y = 48
    draw_number_grid(
        canvas, patch, grid_x, grid_y, cell, cell,
        text_color_fn=lambda v: C_TEXT,
    )

    # --- Column 2: SE mask ---
    x0 = pad + col_w
    draw_title(canvas, f"SE: {se_name}   ({op})", x0, 26, color=C_TITLE)
    grid_x = x0 + (col_w - 2 * pad - k * cell) // 2
    # Show 1s in green, 0s in dark gray
    se_display = se.astype(np.float32)
    def se_color(v):
        return C_GREEN if v > 0 else C_ZERO
    draw_number_grid(canvas, se_display, grid_x, grid_y, cell, cell,
                     text_color_fn=se_color)

    # --- Column 3: masked values (values under the SE footprint) ---
    x0 = pad + 2 * col_w
    draw_title(canvas, "Values under SE", x0, 26, color=C_TITLE)
    # Show the full patch but gray out cells not under the SE
    masked_patch = np.where(se > 0, patch, np.nan).astype(np.float32)
    grid_x = x0 + (col_w - 2 * pad - k * cell) // 2
    # Draw the base patch first (grayed out), then overlay active cells
    for r in range(k):
        for c in range(k):
            cx0 = grid_x + c * cell
            cy0 = grid_y + r * cell
            cx1 = cx0 + cell
            cy1 = cy0 + cell
            if se[r, c] > 0:
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), C_PANEL, -1)
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), C_GREEN, 2)
                txt = f"{int(patch[r, c])}"
                draw_centered_text(canvas, txt,
                                   (cx0 + cx1) // 2, (cy0 + cy1) // 2,
                                   color=C_GREEN, scale=0.5)
            else:
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), C_PANEL, -1)
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), C_BORDER, 1)
                txt = f"{int(patch[r, c])}"
                draw_centered_text(canvas, txt,
                                   (cx0 + cx1) // 2, (cy0 + cy1) // 2,
                                   color=C_ZERO, scale=0.5)

    # --- Column 4: reduction card ---
    x0 = pad + 3 * col_w
    card_x0 = x0
    card_x1 = MATH_W - pad
    card_y0 = 12
    card_y1 = MATH_H - 12

    reduction_name = "max" if op == "dilate" else "min"
    # Build the reduction expression from the actual active values
    vals_str = ", ".join(str(int(v)) for v in active[:12])
    if active.size > 12:
        vals_str += f", ... ({active.size} total)"

    # Header line shows the operator with its formal name
    if op == "dilate":
        header = [("Dilation  (f \u2295 b)", 0.6, C_TITLE)]
    else:
        header = [("Erosion  (f \u2296 b)", 0.6, C_TITLE)]

    body = [
        (f"= {reduction_name}{{", 0.5, C_TEXT),
        (f"   {vals_str}", 0.42, C_TEXT),
        ("  }", 0.5, C_TEXT),
        ("", 0.2, C_TEXT),
        (f"= {int(result)}", 0.75, C_ACCENT),
        ("\u2190 output pixel", 0.45, C_GREEN),
    ]

    draw_equation_card(
        canvas, card_x0, card_y0, card_x1, card_y1,
        header_lines=header,
        body_lines=body,
    )

    return canvas

    # ============================================================================
# SECTION 7 — PANEL 3: OTSU THRESHOLDING WITH LIVE sigma_B^2(T) CURVE
# ============================================================================
# Otsu's method finds the threshold T that maximizes the between-class
# variance of the intensity histogram:
#
#     sigma_B^2(T) = omega_0(T) * omega_1(T) * (mu_0(T) - mu_1(T))^2
#
# where omega_0 and omega_1 are the class probabilities (masses of the
# histogram below/above T) and mu_0, mu_1 are the class means.
#
# The optimum is:
#
#     T* = argmax_T sigma_B^2(T)
#
# We plot:
#   1. The histogram (with T* marked)
#   2. The sigma_B^2(T) curve (with the peak marked)
#   3. An arithmetic card showing omega_0, omega_1, mu_0, mu_1 at T*
#
# Rendering is done in matplotlib and cached as a numpy array. We only
# redraw every MATH_REDRAW_EVERY frames for performance.
# ----------------------------------------------------------------------------

def compute_otsu(gray):
    """
    Vectorized Otsu. Returns (T_star, sigma_B2_curve, hist, mu_T).

    Uses the cumsum trick to evaluate all 256 candidate thresholds at once.
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = int(hist.sum())
    if total == 0:
        return 0, np.zeros(256), hist, 0.0

    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)                              # omega_0(T)
    levels = np.arange(256, dtype=np.float64)
    mu_cum = np.cumsum(levels * p)
    mu_T = mu_cum[-1]

    denom = omega * (1.0 - omega)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    sigma_B2 = (mu_T * omega - mu_cum) ** 2 / denom
    T_star = int(np.argmax(sigma_B2))
    return T_star, sigma_B2, hist, mu_T


def fig_to_bgr(fig):
    """Rasterize a matplotlib figure into a BGR numpy array and close it."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR).copy()
    plt.close(fig)
    return bgr


def render_otsu_panel(gray):
    """Render the Otsu panel as a matplotlib figure, return it as BGR."""
    T_star, sigma_B2, hist, mu_T = compute_otsu(gray)

    fig = plt.figure(figsize=(14, 4.2), dpi=100, facecolor="#121218")
    gs = GridSpec(1, 3, figure=fig, wspace=0.28,
                  left=0.05, right=0.98, top=0.86, bottom=0.14)

    # --- (a) histogram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1a24")
    ax1.bar(np.arange(256), hist, width=1.0, color="#4ea8de")
    ax1.axvline(T_star, color="#ff4757", linestyle="--", linewidth=2)
    ax1.set_title(f"Histogram (mu_T = {mu_T:.1f})",
                  color="#e6e6f0", fontsize=11)
    ax1.set_xlabel("intensity", color="#c8c8d8")
    ax1.tick_params(colors="#c8c8d8")
    for s in ax1.spines.values():
        s.set_color("#5a5a6e")

    # --- (b) sigma_B^2(T) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1a24")
    ax2.plot(sigma_B2, color="#a3e635", linewidth=2)
    ax2.axvline(T_star, color="#ff4757", linestyle="--", linewidth=2)
    ax2.plot(T_star, sigma_B2[T_star], "o", color="#ff4757",
             markersize=11, zorder=5)
    ax2.set_title("sigma_B^2(T)  —  between-class variance",
                  color="#e6e6f0", fontsize=11)
    ax2.set_xlabel("threshold T", color="#c8c8d8")
    ax2.tick_params(colors="#c8c8d8")
    for s in ax2.spines.values():
        s.set_color("#5a5a6e")

    # --- (c) arithmetic card ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1a1a24")
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_color("#5a5a6e")

    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega0 = p[: T_star + 1].sum()
    omega1 = 1.0 - omega0
    levels = np.arange(256, dtype=np.float64)
    mu0 = (levels[: T_star + 1] * p[: T_star + 1]).sum() / max(omega0, 1e-12)
    mu1 = (levels[T_star + 1:] * p[T_star + 1:]).sum() / max(omega1, 1e-12)
    sb2_star = omega0 * omega1 * (mu0 - mu1) ** 2

    ax3.text(0.02, 0.92, "Otsu at T*", color="#ffd166",
             fontsize=13, fontweight="bold", transform=ax3.transAxes)
    ax3.text(0.02, 0.76, f"T* = argmax sigma_B^2(T) = {T_star}",
             color="#e6e6f0", fontsize=11, family="monospace",
             transform=ax3.transAxes)
    ax3.text(0.02, 0.60, f"omega_0 = {omega0:.3f}    mu_0 = {mu0:.2f}",
             color="#a3e635", fontsize=11, family="monospace",
             transform=ax3.transAxes)
    ax3.text(0.02, 0.48, f"omega_1 = {omega1:.3f}    mu_1 = {mu1:.2f}",
             color="#a3e635", fontsize=11, family="monospace",
             transform=ax3.transAxes)
    ax3.text(0.02, 0.32, "sigma_B^2(T*) = omega_0 * omega_1 * (mu_0-mu_1)^2",
             color="#4ea8de", fontsize=10, family="monospace",
             transform=ax3.transAxes)
    ax3.text(0.02, 0.14, f"= {sb2_star:.1f}",
             color="#ff4757", fontsize=18, fontweight="bold",
             transform=ax3.transAxes)

    return fig_to_bgr(fig)


# ============================================================================
# SECTION 8 — PANEL 4: HOUGH ACCUMULATOR
# ============================================================================
# The Hough transform detects lines by voting in (rho, theta) space. Each
# edge pixel (x_i, y_i) votes for every line that passes through it:
#
#     rho = x_i * cos(theta) + y_i * sin(theta)
#
# In the (rho, theta) plane, that equation traces a sinusoid. Where many
# sinusoids cross, many pixels agree on a line — that crossing is the peak
# of the accumulator A(rho, theta).
#
# This panel shows:
#   1. The Canny edge map (source of votes)
#   2. The accumulator A(rho, theta) with the top-3 peaks marked
#   3. The detected lines drawn back on the edge map
# ----------------------------------------------------------------------------

def hough_lines(edges, rho_res=1.0, theta_res=np.pi / 180.0,
                n_peaks=3, min_votes=20):
    """
    Compute a Hough accumulator and return the top-N peaks.

    Returns
    -------
    acc     : accumulator (n_rho, n_theta)
    rhos    : array of rho values
    thetas  : array of theta values
    peaks   : list of (rho, theta, votes)
    """
    h, w = edges.shape
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag, diag + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return acc, rhos, thetas, []

    # Subsample for performance
    if len(xs) > 2000:
        idx = np.random.choice(len(xs), 2000, replace=False)
        xs, ys = xs[idx], ys[idx]

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        rho_idx = np.round((rho_vals + diag) / rho_res).astype(int)
        valid = (rho_idx >= 0) & (rho_idx < len(rhos))
        acc[rho_idx[valid], np.arange(len(thetas))[valid]] += 1

    # Greedy peak selection with local suppression
    acc_work = acc.copy()
    peaks = []
    for _ in range(n_peaks):
        if acc_work.max() < min_votes:
            break
        r_idx, t_idx = np.unravel_index(np.argmax(acc_work), acc_work.shape)
        votes = int(acc_work[r_idx, t_idx])
        peaks.append((float(rhos[r_idx]), float(thetas[t_idx]), votes))
        # Suppress a window around the peak so we don't re-detect it
        r_lo = max(0, r_idx - 20); r_hi = min(acc_work.shape[0], r_idx + 21)
        t_lo = max(0, t_idx - 10); t_hi = min(acc_work.shape[1], t_idx + 11)
        acc_work[r_lo:r_hi, t_lo:t_hi] = 0

    return acc, rhos, thetas, peaks


def render_hough_panel(edges):
    """Render the Hough panel as a matplotlib figure, return it as BGR."""
    acc, rhos, thetas, peaks = hough_lines(edges)

    fig = plt.figure(figsize=(14, 4.2), dpi=100, facecolor="#121218")
    gs = GridSpec(1, 2, figure=fig, wspace=0.22,
                  left=0.05, right=0.98, top=0.86, bottom=0.14)

    # --- (a) edge map with detected lines overlaid ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1a24")
    ax1.imshow(edges, cmap="gray")
    ax1.set_title("Edge map + detected lines", color="#e6e6f0", fontsize=11)
    ax1.set_xticks([]); ax1.set_yticks([])

    h, w = edges.shape
    for (rho, theta, votes) in peaks:
        ct, st = np.cos(theta), np.sin(theta)
        # Two points on the line: x=0 and x=w-1
        x0, y0 = 0, rho / max(st, 1e-9)
        x1, y1 = w - 1, (rho - (w - 1) * ct) / max(st, 1e-9)
        ax1.plot([x0, x1], [y0, y1], "-", color="#ff4757",
                 linewidth=2, alpha=0.85)

    # --- (b) accumulator ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1a24")
    theta_deg = np.degrees(thetas)
    # imshow with extent: origin="lower" to put rho increasing upward
    extent = [theta_deg[0], theta_deg[-1], rhos[0], rhos[-1]]
    ax2.imshow(acc, cmap="inferno", aspect="auto", extent=extent,
               origin="lower")
    ax2.set_xlabel("theta (degrees)", color="#c8c8d8")
    ax2.set_ylabel("rho (pixels)", color="#c8c8d8")
    ax2.set_title("Accumulator A(rho, theta)", color="#e6e6f0", fontsize=11)
    ax2.tick_params(colors="#c8c8d8")
    for s in ax2.spines.values():
        s.set_color("#5a5a6e")

    for i, (rho, theta, votes) in enumerate(peaks):
        ax2.plot(np.degrees(theta), rho, "o", markerfacecolor="none",
                 markeredgecolor="#7dffa0", markeredgewidth=2.5,
                 markersize=14)
        ax2.text(np.degrees(theta) + 4, rho,
                 f"  peak {i+1}: {votes} votes",
                 color="#7dffa0", fontsize=9, fontweight="bold")

    return fig_to_bgr(fig)


# ============================================================================
# SECTION 9 — MOOD DETECTOR: FACE, ROI SPLIT, AU PROXIES
# ============================================================================
# Given a face box, split it into three horizontal bands (brow, eye, mouth)
# and compute classical-CV features from each. Those features are proxies
# for Facial Action Units (FACS):
#
#   AU12  lip corner puller     -> parabola curvature of the mouth contour
#   AU6   cheek raiser          -> inverse of eye aspect ratio (proxy)
#   AU4   brow lowerer          -> negative median Hough angle on brow
#   AU1   inner brow raiser     -> positive median Hough angle on brow
#   AU25  lips part             -> mouth aspect ratio (h / w)
# ----------------------------------------------------------------------------

def load_face_detector(script_dir):
    """
    Try DNN first, fall back to Haar. Returns (kind, detector).

    kind is one of: "dnn", "haar", "none".
    """
    proto = os.path.join(script_dir, "deploy.prototxt")
    model = os.path.join(script_dir,
                         "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        try:
            net = cv2.dnn.readNetFromCaffe(proto, model)
            print(f"[focusguard] Loaded DNN face detector from {script_dir}")
            return "dnn", net
        except cv2.error as e:
            print(f"[focusguard] DNN load failed ({e}); falling back to Haar")
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(haar_path):
        print("[focusguard] Using Haar cascade fallback (DNN files missing)")
        return "haar", cv2.CascadeClassifier(haar_path)
    print("[focusguard] WARNING: no face detector available")
    return "none", None


def detect_face_dnn(net, frame_bgr, conf_thresh=0.5):
    """Return the highest-confidence face box (x1,y1,x2,y2) or None."""
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    det = net.forward()
    best, best_conf = None, conf_thresh
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf > best_conf:
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            best = box.astype(int)
            best_conf = conf
    return best


def detect_face_haar(haar, gray_small):
    """Return the largest face box in the small image or None."""
    boxes = haar.detectMultiScale(gray_small, scaleFactor=1.15,
                                  minNeighbors=5, minSize=(60, 60))
    if len(boxes) == 0:
        return None
    # Largest by area
    areas = [bw * bh for (_, _, bw, bh) in boxes]
    idx = int(np.argmax(areas))
    x, y, bw, bh = boxes[idx]
    return np.array([x, y, x + bw, y + bh], dtype=int)


def mouth_au(gray, mouth_roi, mouth_roi_bgr):
    """
    Otsu on mouth ROI -> morphological opening + closing -> contour fit.

    Returns (au12, au25, mask, T_star, sigma_B2) for visualization.
    """
    if mouth_roi.size == 0:
        return 0.0, 0.0, np.zeros_like(gray), 0, None

    # 1. Otsu threshold on the gray mouth ROI. Since lips are darker than
    #    surrounding skin, use INV so mouth opening (interior) becomes white.
    T_star, sigma_B2, _, _ = compute_otsu(mouth_roi)
    _, mask = cv2.threshold(mouth_roi, T_star, 255, cv2.THRESH_BINARY_INV)

    # 2. Morphological opening (removes lip speckle, salt noise)
    #    opening = erosion followed by dilation: A o B = (A - B) + B
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)

    # 3. Morphological closing (fills small teeth gaps, pepper noise)
    #    closing = dilation followed by erosion: A * B = (A + B) - B
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)

    # 4. Largest contour -> parabolic fit to the top edge
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, mask, T_star, sigma_B2

    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return 0.0, 0.0, mask, T_star, sigma_B2

    pts = c.reshape(-1, 2)
    # Split points into left half and right half; use the upper envelope
    # (smaller y = higher on screen) to fit a parabola that captures smile
    # curvature. We fit y = a x^2 + b x + c using all contour points.
    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)
    try:
        coeffs = np.polyfit(xs, ys, 2)
        a = float(coeffs[0])
    except (np.linalg.LinAlgError, ValueError):
        a = 0.0

    # AU12: parabola curvature. Positive curvature (a > 0) => smile.
    au12 = float(np.clip(-a * 200.0, 0.0, 1.0))

    # AU25: mouth aspect ratio
    _, _, w_box, h_box = cv2.boundingRect(c)
    au25 = float(np.clip(h_box / max(w_box, 1) / 0.6, 0.0, 1.0))

    return au12, au25, mask, T_star, sigma_B2


def eye_au(gray, eye_roi):
    """Adaptive threshold on the eye ROI -> dark pixel ratio -> AU6 proxy."""
    if eye_roi.size == 0:
        return 0.0, np.zeros_like(gray)

    # Adaptive mean threshold (handles changing classroom lighting)
    mask = cv2.adaptiveThreshold(eye_roi, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 5)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)

    # Fraction of dark pixels
    dark_ratio = float((mask > 0).sum()) / max(mask.size, 1)

    # AU6 proxy: eye "squint" = inverse of openness. When the eye is wide
    # open, dark_ratio is high (lots of pupil/iris). When squinting, the
    # lids cover the eye and dark_ratio drops.
    au6 = float(np.clip(1.0 - dark_ratio / 0.25, 0.0, 1.0))
    return au6, mask


def brow_au(gray, brow_roi):
    """Canny + Hough on the brow ROI -> median line angle -> AU1 / AU4."""
    if brow_roi.size == 0:
        return 0.0, 0.0, np.zeros_like(gray), None

    edges = cv2.Canny(brow_roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                            minLineLength=15, maxLineGap=8)
    if lines is None or len(lines) == 0:
        return 0.0, 0.0, edges, None

    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Fold into [0, 90) because brows are roughly horizontal
        ang = abs(ang)
        if ang > 90:
            ang = 180 - ang
        angles.append(ang)

    median_ang = float(np.median(angles))

    # AU1 (inner brow raiser, brow slants up toward center) vs AU4
    # (brow lowerer, brow pushes down toward center). We use the sign
    # of the mean signed angle as the discriminator.
    signed = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        signed.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    mean_signed = float(np.mean(signed))

    au4 = float(np.clip(-mean_signed / 20.0, 0.0, 1.0))   # brow down
    au1 = float(np.clip( mean_signed / 20.0, 0.0, 1.0))   # brow up
    return au1, au4, edges, median_ang


def init_kalman_mood(n=5):
    """5-state Kalman filter, one state per AU, unity transition."""
    kf = cv2.KalmanFilter(n, n)
    kf.transitionMatrix = np.eye(n, dtype=np.float32)
    kf.measurementMatrix = np.eye(n, dtype=np.float32)
    kf.processNoiseCov = np.eye(n, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(n, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(n, dtype=np.float32)
    kf.statePost = np.zeros((n, 1), dtype=np.float32)
    return kf


def classify_mood(au_vec):
    """Rule-based classifier on the smoothed AU vector."""
    au12, au6, au4, au1, au25 = au_vec
    if au12 > 0.35 and au6 > 0.15:
        return "happy"
    if au4 > 0.40 and au1 < 0.20:
        return "focused"
    if au1 > 0.35 and au25 > 0.25:
        return "surprised"
    if au25 > 0.40:
        return "talking"
    return "neutral"


# ============================================================================
# SECTION 10 — LIVENESS: OPTICAL FLOW ON FACE CORNERS
# ============================================================================
# A printed photo of a face is static. A live face has micro-motion: breathing,
# head drift, eye blinks. We measure the mean optical flow magnitude on a
# sparse set of corners inside the face box. If it's below a threshold, we
# declare the input suspicious ("SPOOF?").
# ----------------------------------------------------------------------------

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def liveness_flow(gray, face_box):
    """
    Return mean optical flow magnitude in the face box (float, px/frame).

    State (prev_gray, prev_pts) is stored in STATE and updated each call.
    """
    if face_box is None:
        STATE["prev_gray"] = gray
        STATE["prev_pts"] = None
        return 0.0

    x1, y1, x2, y2 = [int(v) for v in face_box]
    h, w = gray.shape
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return 0.0

    roi_mask = np.zeros_like(gray)
    roi_mask[y1:y2, x1:x2] = 255

    prev_gray = STATE["prev_gray"]
    prev_pts = STATE["prev_pts"]

    if prev_gray is None or prev_pts is None or len(prev_pts) < 5:
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=30,
                                      qualityLevel=0.01, minDistance=10,
                                      mask=roi_mask)
        STATE["prev_gray"] = gray
        STATE["prev_pts"] = pts
        return 0.0

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None, **LK_PARAMS
    )
    STATE["prev_gray"] = gray

    if next_pts is None or status is None:
        STATE["prev_pts"] = None
        return 0.0

    good = status.ravel() == 1
    if good.sum() < 3:
        STATE["prev_pts"] = next_pts
        return 0.0

    flow = np.linalg.norm(next_pts[good] - prev_pts[good], axis=2).mean()
    STATE["prev_pts"] = next_pts
    return float(flow)


# ============================================================================
# SECTION 11 — MOUSE CALLBACK, KEYBOARD, MAIN LOOP
# ============================================================================

def on_mouse(event, x, y, flags, param):
    """Track the mouse position over the FocusGuard window."""
    if event == cv2.EVENT_MOUSEMOVE:
        STATE["mouse"] = (x, y)


def draw_mood_overlay(frame, data):
    """Draw AU bars, mood label, focus score, liveness onto the frame."""
    h, w = frame.shape[:2]

    # AU bars (top-left)
    au_names = ["AU12", "AU6", "AU4", "AU1", "AU25"]
    au_values = data["au"]
    bar_x0, bar_y0 = 12, 12
    bar_max = 160
    for i, (name, val) in enumerate(zip(au_names, au_values)):
        y0 = bar_y0 + i * 22
        y1 = y0 + 16
        # background bar
        cv2.rectangle(frame, (bar_x0, y0), (bar_x0 + bar_max, y1),
                      (40, 40, 52), -1)
        # filled bar
        fill_w = int(bar_max * float(val))
        cv2.rectangle(frame, (bar_x0, y0), (bar_x0 + fill_w, y1),
                      (180, 200, 80), -1)
        cv2.putText(frame, name, (bar_x0 + bar_max + 8, y1 - 2),
                    FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)

    # Mood + focus (top-right)
    mood_text = f"Mood: {data['mood']}"
    focus_text = f"Focus: {data['focus']:.0f}"
    (tw1, _), _ = cv2.getTextSize(mood_text, FONT, 0.7, 2)
    (tw2, _), _ = cv2.getTextSize(focus_text, FONT, 0.7, 2)
    x_mood = w - tw1 - 16
    x_focus = w - tw2 - 16
    cv2.putText(frame, mood_text, (x_mood, 34), FONT, 0.7,
                (80, 220, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, focus_text, (x_focus, 64), FONT, 0.7,
                (240, 200, 80), 2, cv2.LINE_AA)

    # Focus bar
    fb_w = 220
    fb_x0 = w - fb_w - 16
    fb_y0 = 78
    cv2.rectangle(frame, (fb_x0, fb_y0), (fb_x0 + fb_w, fb_y0 + 10),
                  (40, 40, 52), -1)
    fill_w = int(fb_w * float(data["focus"]) / 100.0)
    bar_color = (0, 220, 0) if data["focus"] > 60 else \
                (0, 200, 220) if data["focus"] > 30 else (0, 0, 240)
    cv2.rectangle(frame, (fb_x0, fb_y0), (fb_x0 + fill_w, fb_y0 + 10),
                  bar_color, -1)

    # Liveness (bottom-right)
    if STATE["show_liveness"]:
        live = "REAL" if data["motion"] > 0.15 else "SPOOF?"
        color = (0, 220, 0) if live == "REAL" else (0, 0, 240)
        cv2.putText(frame, live, (w - 120, h - 20), FONT, 0.8,
                    color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"flow={data['motion']:.2f}",
                    (w - 120, h - 46), FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)

    # Face box
    if data["face_box"] is not None:
        x1, y1, x2, y2 = data["face_box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)

    # Cursor crosshair (from mouse position)
    mx, my = STATE["mouse"]
    if 0 <= mx < w and 0 <= my < h:
        cv2.drawMarker(frame, (mx, my), C_YELLOW,
                       cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)


def draw_cursor_box(frame, gray_shape, kernel_size):
    """Draw the k x k neighborhood box around the cursor in the display."""
    mx, my = STATE["mouse"]
    h_frame, w_frame = frame.shape[:2]
    h_gray, w_gray = gray_shape[:2]
    sx = w_frame / max(w_gray, 1)
    sy = h_frame / max(h_gray, 1)
    r_x = int(kernel_size * sx / 2)
    r_y = int(kernel_size * sy / 2)
    cv2.rectangle(frame, (mx - r_x, my - r_y), (mx + r_x, my + r_y),
                  C_GREEN, 1)


def analyze_face(frame_small, gray_small):
    """Run the full mood/focus/liveness pipeline for one frame."""
    # --- face detect (uses cached box between detect frames) ---
    box = STATE["face_box"]
    if box is None:
        box = None

    result = {
        "face_box": box,
        "au": np.zeros(5, dtype=np.float32),
        "mood": "neutral",
        "focus": 0.0,
        "motion": 0.0,
        "mouth_mask": None,
        "eye_mask": None,
        "brow_edges": None,
    }

    if box is None:
        return result

    x1, y1, x2, y2 = [int(v) for v in box]
    h_s, w_s = gray_small.shape[:2]
    x1 = max(0, min(x1, w_s - 1)); x2 = max(0, min(x2, w_s - 1))
    y1 = max(0, min(y1, h_s - 1)); y2 = max(0, min(y2, h_s - 1))
    fh = y2 - y1
    fw = x2 - x1
    if fh < 40 or fw < 40:
        return result

    # ROI split: brow (top 1/4), eye (next 1/4), mouth (bottom 2/5)
    brow_roi = gray_small[y1:y1 + fh // 4, x1:x2]
    eye_roi = gray_small[y1 + fh // 5:y1 + 2 * fh // 5, x1:x2]
    mouth_y0 = y1 + int(fh * 0.6)
    mouth_roi = gray_small[mouth_y0:y2, x1:x2]
    mouth_roi_bgr = frame_small[mouth_y0:y2, x1:x2]

    # --- mouth: Otsu + morphology + contour ---
    au12, au25, mouth_mask, _, _ = mouth_au(gray_small, mouth_roi,
                                             mouth_roi_bgr)
    # --- eye: adaptive threshold ---
    au6, eye_mask = eye_au(gray_small, eye_roi)
    # --- brow: Canny + Hough ---
    au1, au4, brow_edges, _ = brow_au(gray_small, brow_roi)

    au_raw = np.array([au12, au6, au4, au1, au25], dtype=np.float32)

    # --- Kalman smoothing ---
    if STATE["mood_kf"] is None:
        STATE["mood_kf"] = init_kalman_mood(5)
    STATE["mood_kf"].correct(au_raw.reshape(-1, 1).astype(np.float32))
    au_smooth = STATE["mood_kf"].predict().ravel()
    au_smooth = np.clip(au_smooth, 0.0, 1.0)

    # --- liveness ---
    motion = liveness_flow(gray_small, box) if STATE["show_liveness"] else 0.0

    # --- focus score ---
    now = time.time()
    au_eye = au_smooth[1]  # AU6 proxy
    # crude blink detection: eye "closed" when AU6 (squint proxy) > 0.6
    if au_eye > 0.6:
        # record a blink once per event (debounced by 0.25s)
        last = STATE["blink_times"][-1] if STATE["blink_times"] else 0
        if now - last > 0.25:
            STATE["blink_times"].append(now)
    blink_rate = sum(1 for t in STATE["blink_times"] if now - t < 60.0)
    focus = float(np.clip(100.0 - 1.5 * blink_rate - 100.0 * motion,
                          0.0, 100.0))

    result.update({
        "au": au_smooth,
        "mood": classify_mood(au_smooth),
        "focus": focus,
        "motion": motion,
        "mouth_mask": mouth_mask,
        "eye_mask": eye_mask,
        "brow_edges": brow_edges,
    })
    return result


def render_math_panel(gray_small, frame_count):
    """
    Dispatch to the active panel renderer and return the canvas to display.
    Cached matplotlib panels (Otsu, Hough) are redrawn every N frames.
    """
    panel = STATE["panel"]

    if panel == "kernel":
        # Map mouse coords from display size to work size
        kernel_name = KERNEL_ORDER[STATE["kernel_idx"]]
        mx, my = STATE["mouse"]
        # Use the current mouse position directly in the small image space
        # (the mouse callback records display coords; scale down here)
        # Note: the caller passes mouse in small coords via STATE["mouse_small"]
        x_s = STATE.get("mouse_small_x", mx)
        y_s = STATE.get("mouse_small_y", my)
        return render_kernel_panel(gray_small, int(x_s), int(y_s),
                                   kernel_name)

    if panel == "morph":
        se_name = SE_ORDER[STATE["se_idx"]]
        op = MORPH_OPS[STATE["op_idx"]]
        x_s = STATE.get("mouse_small_x", 0)
        y_s = STATE.get("mouse_small_y", 0)
        return render_morph_panel(gray_small, int(x_s), int(y_s),
                                  se_name, op)

    if panel == "otsu":
        if STATE["otsu_cache"] is None or \
                frame_count % MATH_REDRAW_EVERY == 0:
            STATE["otsu_cache"] = render_otsu_panel(gray_small)
        return STATE["otsu_cache"]

    if panel == "hough":
        if STATE["hough_cache"] is None or \
                frame_count % MATH_REDRAW_EVERY == 0:
            edges = cv2.Canny(gray_small, 50, 150)
            STATE["hough_cache"] = render_hough_panel(edges)
        return STATE["hough_cache"]

    return blank_canvas(MATH_W, MATH_H)


def print_banner():
    print("=" * 62)
    print(" FocusGuard running.")
    print(" Controls:")
    print("   1 / 2 / 3 / 4   Switch math panel")
    print("   k               Cycle kernel")
    print("   m               Cycle structuring element")
    print("   o               Toggle dilate <-> erode")
    print("   l               Toggle LAB <-> DEMO")
    print("   s               Toggle liveness detection")
    print("   q               Quit (runs session report)")
    print(" Press the VS Code Stop button (square icon) if 'q' fails.")
    print("=" * 62)


def handle_key(key):
    """Return True to continue, False to quit."""
    if key == 255 or key == -1:
        return True
    if key == ord("q"):
        return False
    if key == ord("1"): STATE["panel"] = "kernel"
    elif key == ord("2"): STATE["panel"] = "morph"
    elif key == ord("3"): STATE["panel"] = "otsu"
    elif key == ord("4"): STATE["panel"] = "hough"
    elif key == ord("k"):
        STATE["kernel_idx"] = (STATE["kernel_idx"] + 1) % len(KERNEL_ORDER)
    elif key == ord("m"):
        STATE["se_idx"] = (STATE["se_idx"] + 1) % len(SE_ORDER)
    elif key == ord("o"):
        STATE["op_idx"] = (STATE["op_idx"] + 1) % len(MORPH_OPS)
    elif key == ord("l"):
        STATE["mode"] = "DEMO" if STATE["mode"] == "LAB" else "LAB"
    elif key == ord("s"):
        STATE["show_liveness"] = not STATE["show_liveness"]
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kind, detector = load_face_detector(script_dir)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[focusguard] ERROR: cannot open camera {CAM_INDEX}")
        return

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MATH, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)

    STATE["start_time"] = time.time()
    print_banner()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[focusguard] camera read failed; exiting loop")
                break

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)
            h_full, w_full = frame.shape[:2]

            # Downscale for CV work
            small = cv2.resize(frame, (WORK_W, WORK_H),
                               interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            STATE["frame_count"] += 1
            fc = STATE["frame_count"]

            # Map mouse coords from display to small
            mx_disp, my_disp = STATE["mouse"]
            mx_small = int(mx_disp * WORK_W / max(w_full, 1))
            my_small = int(my_disp * WORK_H / max(h_full, 1))
            mx_small = int(np.clip(mx_small, 0, WORK_W - 1))
            my_small = int(np.clip(my_small, 0, WORK_H - 1))
            STATE["mouse_small_x"] = mx_small
            STATE["mouse_small_y"] = my_small

            # Face detect every N frames
            if fc % DETECT_EVERY == 0:
                if kind == "dnn":
                    box = detect_face_dnn(detector, small)
                elif kind == "haar":
                    box = detect_face_haar(detector, gray_small)
                else:
                    box = None
                # Scale box back up to full frame for overlay drawing
                if box is not None:
                    sx = w_full / WORK_W
                    sy = h_full / WORK_H
                    STATE["face_box"] = np.array([
                        int(box[0] * sx), int(box[1] * sy),
                        int(box[2] * sx), int(box[3] * sy),
                    ], dtype=int)
                else:
                    STATE["face_box"] = None

            # Analyze face using the SMALL frame (fast), but pass the
            # full-size frame for the box (drawn by overlay)
            small_box = None
            if STATE["face_box"] is not None:
                sx = WORK_W / w_full
                sy = WORK_H / h_full
                b = STATE["face_box"]
                small_box = np.array([
                    int(b[0] * sx), int(b[1] * sy),
                    int(b[2] * sx), int(b[3] * sy),
                ], dtype=int)
            STATE["face_box"] = STATE["face_box"]  # keep full-size for draw
            _saved = STATE["face_box"]
            # Temporarily swap in small box for analysis
            STATE["face_box"] = small_box
            mood_data = analyze_face(small, gray_small)
            STATE["face_box"] = _saved
            mood_data["face_box"] = _saved

            # Draw overlay on full-size frame
            disp = frame.copy()
            draw_mood_overlay(disp, mood_data)

            # Neighborhood box around cursor
            if STATE["panel"] == "kernel":
                ksize = KERNELS[KERNEL_ORDER[STATE["kernel_idx"]]].shape[0]
                draw_cursor_box(disp, gray_small.shape, ksize)
            elif STATE["panel"] == "morph":
                ksize = SE_KERNELS[SE_ORDER[STATE["se_idx"]]].shape[0]
                draw_cursor_box(disp, gray_small.shape, ksize)

            # Math panel
            math_canvas = render_math_panel(gray_small, fc)
            math_display = resize_panel(math_canvas, MATH_W, MATH_H)

            # Mode banner (top of math panel)
            banner = f"{STATE['mode']} MODE"
            if STATE["mode"] == "DEMO" and _saved is not None:
                banner += "  —  tracking face ROI"
            banner_color = C_RED if STATE["mode"] == "DEMO" else C_GREEN
            cv2.rectangle(math_display, (0, 0), (MATH_W, 22),
                          (12, 12, 18), -1)
            cv2.putText(math_display, banner, (10, 16), FONT, 0.5,
                        banner_color, 1, cv2.LINE_AA)
            # Panel label (right side of banner)
            label = f"Panel: {STATE['panel']}"
            (lw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.putText(math_display, label, (MATH_W - lw - 10, 16),
                        FONT, 0.5, C_TEXT, 1, cv2.LINE_AA)

            cv2.imshow(WINDOW_MAIN, disp)
            cv2.imshow(WINDOW_MATH, math_display)

            # Log focus for the session report
            if STATE["start_time"] is not None:
                t = time.time() - STATE["start_time"]
                STATE["focus_log"].append((t, mood_data["focus"]))

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if not handle_key(key):
                break

            # Window-closed detection
            if cv2.getWindowProperty(WINDOW_MAIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(WINDOW_MATH, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("[focusguard] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[focusguard] Camera released. Windows closed.")


# ============================================================================
# SECTION 12 — SESSION REPORT
# ============================================================================
# When the user quits with 'q', produce a summary PNG showing the focus
# timeline. Saved next to the script as session_report.png.
# ----------------------------------------------------------------------------

def session_report(output_path="session_report.png"):
    log = STATE["focus_log"]
    if not log:
        print("[focusguard] No session data to report.")
        return None

    times = [t for (t, _) in log]
    scores = [s for (_, s) in log]

    fig = plt.figure(figsize=(11, 3.5), dpi=110, facecolor="#121218")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#1a1a24")
    ax.plot(times, scores, color="#a3e635", linewidth=2)
    ax.fill_between(times, scores, alpha=0.15, color="#a3e635")
    ax.set_xlabel("session time (s)", color="#c8c8d8")
    ax.set_ylabel("focus score", color="#c8c8d8")
    avg = float(np.mean(scores)) if scores else 0.0
    ax.set_title(f"FocusGuard session — avg focus = {avg:.1f}",
                 color="#e6e6f0", fontsize=12)
    ax.tick_params(colors="#c8c8d8")
    ax.grid(True, alpha=0.15, color="#5a5a6e")
    for s in ax.spines.values():
        s.set_color("#5a5a6e")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(output_path, facecolor="#121218")
    plt.close(fig)
    print(f"[focusguard] Session report saved to {output_path}")
    return output_path


# ============================================================================
# SECTION 13 — ENTRY POINT
# ============================================================================
# Note: this block is what makes `python focusguard.py` work standalone.
# When the notebook does `%run focusguard.py`, this block runs too — same
# behavior.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    finally:
        session_report()