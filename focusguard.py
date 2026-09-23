#!/usr/bin/env python3
"""
FocusGuard — Live Classical Computer Vision Demo
================================================

PURPOSE
-------
A teaching demo that shows classical CV math running live on a webcam.
The demo has TWO windows that work together:

  1. FocusGuard  — a "product": reads your facial expressions and
                   computes a mood + focus score. This is what a real
                   app would show a user.

  2. Math Panel  — a "glass box": shows the exact pixel-level arithmetic
                   the product is running underneath (convolution,
                   morphology, Otsu, Hough). The audience can watch a
                   single pixel's math happen in real time.

WHY TWO WINDOWS?
----------------
The pedagogical point: "computer vision is not a black box — it's
arithmetic on pixels, at 30 fps". The product window gives context;
the math window gives proof. Both share an accent color per panel so
viewers see they are one connected system.

ARCHITECTURE OVERVIEW
---------------------
  Webcam frame (mirrored)
        │
        ├──► downscale to 480×360 for CV work (speed)
        │
        ├──► FACE DETECTOR (DNN or Haar fallback)
        │         │
        │         └──► face box (in 480×360 coords)
        │
        ├──► MOOD PIPELINE  (uses face box to crop 3 ROIs)
        │       ├── Mouth ROI → Otsu → morphology → contour → AU12, AU25
        │       ├── Eye ROI   → adaptive threshold → AU6
        │       └── Brow ROI  → Canny + Hough → AU1, AU4
        │
        ├──► Kalman filter on the 5-D AU vector (temporal smoothing)
        │
        ├──► Liveness via optical flow (real face moves, photos don't)
        │
        └──► RENDER
              ├──► FocusGuard canvas (video + overlay + AU bars)
              └──► Math Panel canvas (4 selectable panels)

KEYBOARD CONTROLS
-----------------
  1 / 2 / 3 / 4   Switch math panel (Kernel / Morphology / Otsu / Hough)
  k               Cycle kernel (Sobel X/Y, Prewitt, Roberts, etc.)
  m               Cycle structuring element (square, cross, ellipse, ...)
  o               Toggle dilate <-> erode
  l               Toggle LAB mode <-> DEMO mode
  s               Toggle liveness display
  q               Quit (also writes a session report PNG)

DEPENDENCIES
------------
  pip install opencv-contrib-python numpy matplotlib

Runs on Debian/Linux/macOS/Windows with a webcam at /dev/video0 (Linux).
"""

# ============================================================================
# SECTION 1 — IMPORTS AND ENVIRONMENT SETUP
# ============================================================================
# Two critical setup decisions here:
#
# 1. `matplotlib.use("Agg")` MUST come before importing pyplot. Reason:
#    the math panel figures are rasterized to numpy arrays and blitted
#    into cv2 windows. If matplotlib picks an interactive backend (Qt/
#    Tk), it fights cv2 for the GUI event loop and the math panels
#    render blank. Agg is headless and rasterizes reliably.
#
# 2. Only 4 modules are imported from the stdlib: os, time, shutil,
#    subprocess, collections. These cover path handling, timing,
#    wmctrl detection, subprocess launching (for wmctrl), and a
#    ring buffer for blink timestamps.
# ----------------------------------------------------------------------------

import os               # path operations
import time             # timing (blink timestamps, session duration)
import shutil           # shutil.which() to detect wmctrl
import subprocess       # Popen for optional wmctrl "always on top"
import collections      # deque for a fixed-size ring of blink times

import cv2              # OpenCV — the entire classical CV toolkit
import numpy as np      # array math (kernels, histograms, etc.)

# Agg backend selection MUST precede `import matplotlib.pyplot`
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec   # for the Otsu/Hough subplots


# ============================================================================
# SECTION 2 — CONSTANTS
# ============================================================================
# Everything tunable lives here so behavior can be adjusted without
# hunting through the code. Grouped by concern:
#   - Window names (used by cv2.imshow / moveWindow / getWindowProperty)
#   - Camera and processing sizes
#   - Frame-skip intervals (performance knobs)
#   - Colors (BGR — OpenCV convention, NOT RGB — this trips people up)
#   - Fonts
#   - Kernel and structuring element libraries (the "slides" content)
#   - AU metadata (friendly names for the on-screen display)
#   - Global mutable STATE dict (updated by mouse/keyboard callbacks)
# ----------------------------------------------------------------------------

# Window titles — used consistently so cv2 can find them
WINDOW_MAIN = "FocusGuard"
WINDOW_MATH = "Math Panel"

# Camera configuration
CAM_INDEX = 0                   # /dev/video0 on Linux

# ALL classical CV runs at this size for speed. 480×360 is 4:3, matches
# most webcams after cropping, and is small enough that Sobel + morphology
# + Otsu all run at 30+ fps on a laptop.
WORK_W, WORK_H = 480, 360

# Frame-skip intervals — trade latency for CPU
DETECT_EVERY = 5                # run the face DNN every Nth frame
MATH_REDRAW_EVERY = 8           # redraw matplotlib panels every Nth frame

# ----------------------------------------------------------------------------
# COLOR PALETTE (BGR — OpenCV!)
# ----------------------------------------------------------------------------
# In OpenCV, colors are tuples of (Blue, Green, Red), 0-255 each.
# So (255, 0, 0) is pure blue and (0, 0, 255) is pure red.
# Every color below is defined once and reused — no magic tuples.
# ----------------------------------------------------------------------------
C_BG       = (24, 20, 18)       # very dark warm gray — panel background
C_PANEL    = (36, 32, 30)       # slightly lighter — cell fill
C_PANEL_HI = (56, 48, 44)       # highlighted cell (under SE footprint)
C_BORDER   = (110, 90, 90)      # cell / panel borders
C_TEXT     = (240, 232, 232)    # bright off-white for body text
C_MUTED    = (168, 148, 148)    # subdued gray for secondary text
C_POS      = (160, 255, 125)    # green — positive values
C_NEG      = (107, 107, 255)    # red — negative values
C_ZERO     = (140, 140, 140)    # gray — zeros
C_AMBER    = (102, 209, 255)    # amber — highlights, badges, active state
C_PURPLE   = (255, 125, 199)    # purple — "talking" mood
C_BLUE     = (222, 168, 78)     # blue — "focused" mood / kernel panel
C_GREEN    = (160, 255, 125)    # bright green — for successful detections
C_RED      = (107, 107, 255)    # bright red — for spoof / errors
C_YELLOW   = (0, 220, 220)      # yellow — cursor crosshair

# ----------------------------------------------------------------------------
# FONTS
# ----------------------------------------------------------------------------
# Two fonts are used deliberately:
#   - HERSHEY_TRIPLEX for headers — cleaner letterforms, more polished
#   - HERSHEY_SIMPLEX for numbers and body text — legible at small sizes
# ----------------------------------------------------------------------------
F_TITLE = cv2.FONT_HERSHEY_TRIPLEX
F_BODY  = cv2.FONT_HERSHEY_SIMPLEX

# ----------------------------------------------------------------------------
# PANEL METADATA — used to link the two windows visually
# ----------------------------------------------------------------------------
# Each math panel has an accent color. When you press 1/2/3/4 on the
# FocusGuard window, the SAME color appears on both windows — the
# audience immediately sees the two windows are one system.
# ----------------------------------------------------------------------------
PANEL_ACCENT = {
    "kernel": C_BLUE,       # panel 1: kernel convolution → blue
    "morph":  C_GREEN,      # panel 2: morphology         → green
    "otsu":   C_AMBER,      # panel 3: Otsu thresholding  → amber
    "hough":  C_PURPLE,     # panel 4: Hough transform    → purple
}
PANEL_NAMES = {
    "kernel": "Kernel Convolution",
    "morph":  "Mathematical Morphology",
    "otsu":   "Otsu Thresholding",
    "hough":  "Hough Transform",
}
PANEL_KEY = {"kernel": "1", "morph": "2", "otsu": "3", "hough": "4"}

# ----------------------------------------------------------------------------
# KERNEL LIBRARY
# ----------------------------------------------------------------------------
# Each kernel is a small matrix that slides over the image. At each
# position, we do elementwise multiplication between the kernel and the
# image patch underneath, then sum. That sum is the output pixel value.
#
# In math terms:
#     G(x, y) = sum_u sum_v  I(x+u, y+v) * K(u, v)
#
# Notes on each:
#   sobel_x/y   — first derivative + binomial smoothing. Default choice.
#   prewitt_x/y — first derivative with uniform smoothing (noisier).
#   roberts_x/y — 2×2 diagonal differences (fastest, noisiest).
#   laplacian   — second derivative (edge detection via zero crossings).
#   gaussian    — pure smoothing (discrete approximation of Gaussian).
#   box_blur    — pure averaging (uniform smoothing).
# ----------------------------------------------------------------------------
KERNELS = {
    "sobel_x":   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "sobel_y":   np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "prewitt_x": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "prewitt_y": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "roberts_x": np.array([[1, 0], [0, -1]], dtype=np.float32),   # 2×2!
    "roberts_y": np.array([[0, 1], [-1, 0]], dtype=np.float32),
    "laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
    "gaussian":  np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0,
    "box_blur":  np.ones((3, 3), dtype=np.float32) / 9.0,
}
KERNEL_ORDER = list(KERNELS.keys())     # order for the 'k' key cycle

# ----------------------------------------------------------------------------
# STRUCTURING ELEMENT (SE) LIBRARY — for morphology
# ----------------------------------------------------------------------------
# An SE is a binary mask that says WHICH pixels around a given pixel
# participate in the operation. Only cells where the SE is 1 are used.
#
# Morphological operations:
#   Dilate: (f ⊕ b)(x) = max_{h in B} f(x + h)   [local max]
#   Erode:  (f ⊖ b)(x) = min_{h in B} f(x + h)   [local min]
#
# The shape of the SE matters:
#   square  — isotropic, general purpose
#   cross   — preserves connectivity (4-connected thinning)
#   ellipse — rotationally invariant (no orientation bias)
#   line    — directional (detects horizontal/vertical structures)
#   diag    — diagonal-only (e.g., for skeletonization)
# ----------------------------------------------------------------------------
SE_KERNELS = {
    "square_3":  cv2.getStructuringElement(cv2.MORPH_RECT,    (3, 3)),
    "cross_3":   cv2.getStructuringElement(cv2.MORPH_CROSS,   (3, 3)),
    "ellipse_5": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    "hline_1x9": cv2.getStructuringElement(cv2.MORPH_RECT,    (9, 1)),
    "vline_9x1": cv2.getStructuringElement(cv2.MORPH_RECT,    (1, 9)),
    "diag_5":    np.eye(5, dtype=np.uint8),   # 5×5 identity = diagonal line
}
SE_ORDER = list(SE_KERNELS.keys())
MORPH_OPS = ["dilate", "erode"]

# ----------------------------------------------------------------------------
# FACIAL ACTION UNIT (AU) METADATA
# ----------------------------------------------------------------------------
# FACS (Facial Action Coding System) defines atomic facial muscle
# movements called Action Units. We compute 5 AU proxies from classical
# CV, and display them with a FRIENDLY name so non-experts understand:
#
#   AU12 (Lip corner puller)  → "Smile"       — parabola curvature of mouth
#   AU6  (Cheek raiser)       → "Cheek raise" — eye squint / dark ratio
#   AU4  (Brow lowerer)       → "Brow furrow" — negative Hough angle
#   AU1  (Inner brow raiser)  → "Brow raise"  — positive Hough angle
#   AU25 (Lips part)          → "Mouth open"  — mouth aspect ratio
#
# Display order matters — it matches the order of the AU vector built
# in analyze_face(). DO NOT reorder without also reordering that vector.
# ----------------------------------------------------------------------------
AU_INFO = [
    ("Smile",       "AU12"),
    ("Cheek raise", "AU6"),
    ("Brow furrow", "AU4"),
    ("Brow raise",  "AU1"),
    ("Mouth open",  "AU25"),
]

# ----------------------------------------------------------------------------
# GLOBAL STATE DICT
# ----------------------------------------------------------------------------
# All mutable runtime state lives here. This is an alternative to
# globals: cv2 mouse/keyboard callbacks need to mutate shared state,
# and passing a dict reference is the cleanest way in Python.
# ----------------------------------------------------------------------------
STATE = {
    # Mouse position in the FocusGuard window (used for pixel inspection)
    "mouse": (0, 0),
    "mouse_small_x": 0, "mouse_small_y": 0,   # same, in 480×360 coords

    # Active math panel and per-panel selection indices
    "panel": "kernel",
    "kernel_idx": 0, "se_idx": 0, "op_idx": 0,

    # UI toggles
    "mode": "LAB",              # LAB (free explore) or DEMO (follows face)
    "show_liveness": True,

    # Frame counter (used for % N skip logic)
    "frame_count": 0,

    # Cache for matplotlib-rendered panels (avoid re-rendering every frame)
    "otsu_cache": None, "hough_cache": None,

    # Cache for the kernel/morph panels, keyed by a "signature" tuple
    # (panel, kernel_idx, se_idx, op_idx, mouse_x, mouse_y). If the
    # signature hasn't changed, we reuse the cached canvas.
    "math_cache": None, "math_sig": None,

    # Session tracking for the report at quit time
    "focus_log": [],            # list of (t_sec, focus_score)
    "start_time": None,
    "blink_times": collections.deque(maxlen=600),   # ring buffer

    # Kalman filter for temporal smoothing of the 5-D AU vector
    "mood_kf": None,            # lazily initialized on first use

    # Optical flow state (prev frame + prev corner points)
    "prev_gray": None, "prev_pts": None,

    # Current detected face box in 480×360 coords (or None)
    "face_box": None,
}


# ============================================================================
# SECTION 3 — DRAW UTILITIES
# ============================================================================
# Panels 1 and 2 (kernel, morphology) are drawn ENTIRELY with cv2
# primitives — not matplotlib. Reason: matplotlib re-rendering every
# frame drops fps to ~3. cv2.line/rectangle/putText is 100× faster.
#
# Panels 3 and 4 (Otsu, Hough) use matplotlib because they need
# proper axes, heatmaps and plots. They are cached (rendered once every
# MATH_REDRAW_EVERY frames) and blitted as numpy arrays.
# ----------------------------------------------------------------------------

def blank_canvas(w, h, bg=C_BG):
    """Create a fresh BGR canvas of the given size, filled with a solid color."""
    c = np.zeros((h, w, 3), dtype=np.uint8)
    c[:] = bg
    return c


def draw_badge(canvas, x, y, num, radius=22, active=True):
    """
    Draw a numbered circle badge (①②③④) used as the section header icon.

    active=True  → amber (indicates this is the currently selected panel)
    active=False → dim gray (inactive / informational)
    """
    bg_col = C_AMBER if active else (70, 60, 55)
    cv2.circle(canvas, (x, y), radius, bg_col, -1, cv2.LINE_AA)      # filled
    cv2.circle(canvas, (x, y), radius, C_TEXT, 2, cv2.LINE_AA)       # outline
    txt = str(num)
    # Center the digit inside the circle
    (tw, th), _ = cv2.getTextSize(txt, F_BODY, 1.0, 2)
    cv2.putText(canvas, txt, (x - tw // 2, y + th // 2),
                F_BODY, 1.0, (20, 20, 20), 2, cv2.LINE_AA)


def draw_centered(canvas, text, cx, cy, color=C_TEXT, scale=0.7,
                  font=F_BODY, thickness=1):
    """Draw text horizontally AND vertically centered on (cx, cy)."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, (cx - tw // 2, cy + th // 2),
                font, scale, color, thickness, cv2.LINE_AA)


def draw_number_grid(canvas, values, x0, y0, cw, ch,
                     text_color_fn=None, border_color=C_BORDER,
                     bg_fn=None, strike_fn=None, text_scale=0.85):
    """
    Draw a 2D array of numbers as a grid of cells.

    values          : 2D numpy array (int or float)
    x0, y0          : top-left corner of the grid
    cw, ch          : cell width/height in pixels
    text_color_fn   : optional callable(value) -> BGR color for the text
    border_color    : color of cell borders
    bg_fn           : optional callable(r, c) -> BGR color for cell fill
    strike_fn       : optional callable(r, c) -> bool; if True, draw a
                      diagonal strikethrough (used to mark excluded cells)
    text_scale      : font scale for cell values
    """
    rows, cols = values.shape
    for r in range(rows):
        for c in range(cols):
            cx0, cy0 = x0 + c * cw, y0 + r * ch
            cx1, cy1 = cx0 + cw, cy0 + ch

            # Cell background (default C_PANEL, or custom via bg_fn)
            bg = bg_fn(r, c) if bg_fn else C_PANEL
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), bg, -1)

            # Cell border (2px so it's visible on projector)
            cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), border_color, 2)

            # Optional strikethrough diagonal
            if strike_fn and strike_fn(r, c):
                cv2.line(canvas, (cx0 + 5, cy0 + 5), (cx1 - 5, cy1 - 5),
                         C_ZERO, 2, cv2.LINE_AA)

            # Format the value: integers stay as int, floats get 2 decimals
            v = values[r, c]
            if isinstance(v, (float, np.floating)) and v != int(v):
                txt = f"{v:.2f}".rstrip("0").rstrip(".")
            else:
                txt = str(int(v))

            color = text_color_fn(v) if text_color_fn else C_TEXT
            draw_centered(canvas, txt, (cx0 + cx1) // 2, (cy0 + cy1) // 2,
                          color=color, scale=text_scale, thickness=2)


def draw_legend(canvas):
    """
    Draw the bottom legend strip on math panels.
    Explains the color coding and the keyboard controls — so the demo
    is self-documenting: the audience can read the controls off screen.
    """
    w, h = canvas.shape[1], canvas.shape[0]
    y0 = h - 56                      # strip is 56px tall at the bottom

    # Strip background
    cv2.rectangle(canvas, (0, y0), (w, h), (14, 12, 12), -1)
    cv2.line(canvas, (0, y0), (w, y0), C_BORDER, 1)

    x, ymid = 20, y0 + 28

    def chip(color, label):
        """Draw a colored square + text label, advancing x."""
        nonlocal x
        cv2.rectangle(canvas, (x, ymid - 9), (x + 18, ymid + 9), color, -1)
        x += 26
        cv2.putText(canvas, label, (x, ymid + 7), F_BODY, 0.55,
                    C_TEXT, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(label, F_BODY, 0.55, 1)
        x += tw + 24

    # Color legend
    chip(C_POS, "positive")
    chip(C_NEG, "negative")
    chip(C_ZERO, "zero")

    # Cursor hint
    cv2.drawMarker(canvas, (x + 9, ymid), C_YELLOW,
                   cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    x += 26
    cv2.putText(canvas, "mouse picks pixel", (x, ymid + 7),
                F_BODY, 0.55, C_TEXT, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("mouse picks pixel", F_BODY, 0.55, 1)
    x += tw + 24

    # Keyboard controls
    cv2.putText(canvas,
                "[1-4] panel  [k] kernel  [m] SE  [o] dil/ero  "
                "[l] LAB/DEMO  [q] quit",
                (x, ymid + 7), F_BODY, 0.55, C_AMBER, 1, cv2.LINE_AA)


def draw_section_header(canvas, x0, y0, col_w, num, title, subtitle,
                        active=True):
    """
    Draw a numbered section header (badge + title + subtitle).

    y0 is the TOP of the section, not the baseline. Badge sits 30px down,
    title baseline at y0+40, subtitle at y0+78. This offset pattern is
    reused everywhere so sections align consistently.
    """
    draw_badge(canvas, x0 + 34, y0 + 30, num, radius=24, active=active)
    cv2.putText(canvas, title, (x0 + 70, y0 + 40),
                F_TITLE, 0.85, C_TEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (x0 + 18, y0 + 78),
                F_BODY, 0.55, C_MUTED, 1, cv2.LINE_AA)


# ============================================================================
# SECTION 4 — PATCH EXTRACTION
# ============================================================================
# The single most reused helper. Extracts a k×k neighborhood centered at
# (x, y) from a grayscale image. Out-of-bounds pixels are treated as 0
# (zero padding), which is what cv2.filter2D does at image edges by default.
#
# Why zero padding? Because it produces no boundary artifacts in the
# convolution sum — the kernel just sees black pixels outside the image.
# Alternative schemes (replicate, reflect, wrap) exist but complicate the
# code without pedagogical benefit for this demo.
# ----------------------------------------------------------------------------

def extract_patch(gray, x, y, k):
    """
    Return the k×k neighborhood centered at (x, y), zero-padded at edges.

    For k=3, the patch is laid out as:
        [ I(x-1, y-1)  I(x, y-1)  I(x+1, y-1) ]
        [ I(x-1, y  )  I(x, y  )  I(x+1, y  ) ]
        [ I(x-1, y+1)  I(x, y+1)  I(x+1, y+1) ]

    where I(.) is 0 outside the image bounds.
    """
    h, w = gray.shape
    r = k // 2                         # half-window radius
    patch = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            yy = y + i - r
            xx = x + j - r
            if 0 <= yy < h and 0 <= xx < w:
                patch[i, j] = gray[yy, xx]
    return patch


# ============================================================================
# SECTION 5 — PANEL 1: KERNEL CONVOLUTION
# ============================================================================
# This panel makes the convolution operation visually obvious.
#
# The concept:
#     G(x, y) = sum_u sum_v  I(x+u, y+v) * K(u, v)
#
# We show 4 quadrants for one chosen pixel:
#   ┌────────────────┬────────────────┐
#   │ ① NEIGHBORHOOD │ ② KERNEL       │  ← input data
#   ├────────────────┼────────────────┤
#   │ ③ PRODUCTS     │ ④ RESULT       │  ← arithmetic
#   └────────────────┴────────────────┘
#
# Column ④ shows the actual numbers substituted into the equation, e.g.
#     G(x,y) = -188.1 + 187.1 - 186.2 + ... = 6
# So the audience literally sees the arithmetic happening.
# ----------------------------------------------------------------------------

def convolve_at(gray, x, y, kernel):
    """
    Compute the convolution result at (x, y).

    Returns
    -------
    patch    : k×k neighborhood extracted from the image
    products : k×k elementwise product of patch and kernel
    raw_sum  : the raw scalar sum (can be negative or >255)
    clipped  : the value that would be stored in a uint8 image
               (absolute value, clamped to [0, 255])
    """
    k = kernel.shape[0]
    patch = extract_patch(gray, x, y, k)
    products = patch * kernel
    raw_sum = float(products.sum())
    clipped = int(np.clip(abs(raw_sum), 0, 255))
    return patch, products, raw_sum, clipped


def render_kernel_panel(gray, x, y, kernel_name, cw, ch):
    """
    Build the full Panel 1 (Kernel Convolution) canvas.

    Layout (2×2 grid of sections):
      ┌─────────────────────┬─────────────────────┐
      │ ① PIXEL NEIGHBORHOOD│ ② KERNEL            │
      │ ③ PRODUCTS          │ ④ RESULT (equation) │
      └─────────────────────┴─────────────────────┘
      [ legend strip at bottom ]

    TOP=44 reserves space for the "LINKED TO FOCUSGUARD" banner that
    decorate_math_panel() draws on top afterward.
    """
    kernel = KERNELS[kernel_name]
    k = kernel.shape[0]
    patch, products, raw_sum, clipped = convolve_at(gray, x, y, kernel)

    # Create the base canvas and work out the geometry
    canvas = blank_canvas(cw, ch)
    TOP = 44                                     # clearance for link banner
    legend_h = 56
    body_h = ch - TOP - legend_h
    half_w = cw // 2
    half_h = body_h // 2

    # Cell size: fit the k×k grid into each quadrant with comfortable margin
    avail_w = half_w - 40
    avail_h = half_h - 100                       # 100 for header + spacing
    cell = min(avail_w // max(k, 1), avail_h // max(k, 1), 90)

    # Color function for kernel/product cells: green +, red -, gray 0
    def kcolor(v):
        if v > 0: return C_POS
        if v < 0: return C_NEG
        return C_ZERO

    # --- TOP-LEFT: neighborhood ---
    draw_section_header(canvas, 0, TOP, half_w, 1, "PIXEL NEIGHBORHOOD",
                        f"The {k}x{k} pixels around ({x}, {y})")
    gx = (half_w - k * cell) // 2                # horizontal center
    gy = TOP + 100
    draw_number_grid(canvas, patch, gx, gy, cell, cell,
                     text_color_fn=lambda v: C_POS if v > 128 else C_AMBER,
                     text_scale=1.0)

    # --- TOP-RIGHT: kernel ---
    draw_section_header(canvas, half_w, TOP, half_w, 2, "KERNEL",
                        f"The operator '{kernel_name}' applied to it")
    gx = half_w + (half_w - k * cell) // 2
    gy = TOP + 100
    draw_number_grid(canvas, kernel, gx, gy, cell, cell,
                     text_color_fn=kcolor, border_color=C_AMBER,
                     text_scale=1.0)

    # --- BOTTOM-LEFT: elementwise products ---
    draw_section_header(canvas, 0, TOP + half_h, half_w, 3, "PRODUCTS",
                        "Multiply each cell pair")
    gx = (half_w - k * cell) // 2
    gy = TOP + half_h + 100
    draw_number_grid(canvas, products, gx, gy, cell, cell,
                     text_color_fn=kcolor, text_scale=1.0)

    # --- BOTTOM-RIGHT: the equation with real numbers ---
    draw_section_header(canvas, half_w, TOP + half_h, half_w, 4, "RESULT",
                        "Sum all products -> output pixel")

    eq_x = half_w + 30
    eq_y = TOP + half_h + 130

    # Header line of the equation
    cv2.putText(canvas, "G(x,y) = Sum Sum I*K",
                (eq_x, eq_y), F_TITLE, 0.85, C_AMBER, 2, cv2.LINE_AA)
    eq_y += 44

    # Build the equation string with the ACTUAL numbers from this pixel.
    # Skip zero kernel coefficients — they contribute nothing and clutter.
    terms = []
    for i in range(k):
        for j in range(k):
            c = kernel[i, j]
            if c == 0:
                continue
            s = "+" if c > 0 else "-"
            cc = abs(int(c)) if c == int(c) else abs(c)
            terms.append(f"{s} {int(patch[i,j])}.{cc}")
    expr = " ".join(terms)
    if expr.startswith("+ "):
        expr = expr[2:]                          # remove leading "+ "

    # Wrap the equation to ~32 chars per line, max 3 lines
    lines, cur = [], ""
    for tok in expr.split(" "):
        if len(cur) + len(tok) + 1 > 32:
            lines.append(cur)
            cur = tok
        else:
            cur = (cur + " " + tok) if cur else tok
    if cur:
        lines.append(cur)
    for line in lines[:3]:
        cv2.putText(canvas, line, (eq_x, eq_y), F_BODY, 0.6,
                    C_TEXT, 1, cv2.LINE_AA)
        eq_y += 30

    # The big final answer
    eq_y += 12
    cv2.putText(canvas, f"sum = {int(raw_sum)}",
                (eq_x, eq_y), F_TITLE, 1.4, C_POS, 3, cv2.LINE_AA)
    eq_y += 52
    cv2.putText(canvas, f"clip(|sum|) = {clipped}",
                (eq_x, eq_y), F_BODY, 0.6, C_MUTED, 1, cv2.LINE_AA)
    eq_y += 32
    cv2.putText(canvas, "-> output pixel",
                (eq_x, eq_y), F_BODY, 0.6, C_GREEN, 1, cv2.LINE_AA)

    # Draw dividers AFTER content so they sit on top of anything spilling over
    cv2.line(canvas, (half_w, TOP), (half_w, TOP + body_h), C_BORDER, 1)
    cv2.line(canvas, (0, TOP + half_h), (cw, TOP + half_h), C_BORDER, 1)

    # Legend
    draw_legend(canvas)
    return canvas


# ============================================================================
# SECTION 6 — PANEL 2: MATHEMATICAL MORPHOLOGY
# ============================================================================
# Same 4-quadrant layout as Panel 1, but showing morphological operations.
#
# The concept:
#     Dilate: (f ⊕ b)(x) = max_{h in B} f(x + h)
#     Erode:  (f ⊖ b)(x) = min_{h in B} f(x + h)
#
# For each pixel, look at the k×k neighborhood. Take the MAX (dilate)
# or MIN (erode) of only the values under the SE's 1s.
#
# The visual trick: cells NOT under the SE footprint are grayed out and
# stricken through, so the audience sees which values participate.
# ----------------------------------------------------------------------------

def morph_at(gray, x, y, se, op):
    """
    Compute the morphological operation at (x, y).

    Returns
    -------
    patch  : k×k neighborhood
    active : 1-D array of values under the SE footprint (in scan order)
    result : max (dilate) or min (erode) of those active values
    """
    k = se.shape[0]
    patch = extract_patch(gray, x, y, k)
    mask = se > 0                          # boolean: which cells are active
    active = patch[mask]                   # extract only those values
    if op == "dilate":
        result = float(active.max()) if active.size else 0.0
    else:
        result = float(active.min()) if active.size else 0.0
    return patch, active, result


def render_morph_panel(gray, x, y, se_name, op, cw, ch):
    """
    Build the full Panel 2 (Morphology) canvas.

    Sections:
      ┌─────────────────────┬─────────────────────┐
      │ ① PIXEL NEIGHBORHOOD│ ② STRUCTURING ELEM. │
      │ ③ ACTIVE VALUES     │ ④ REDUCTION (max/min│
      └─────────────────────┴─────────────────────┘
    """
    se = SE_KERNELS[se_name]
    k = se.shape[0]
    patch, active, result = morph_at(gray, x, y, se, op)

    canvas = blank_canvas(cw, ch)
    TOP = 44                                     # clearance for link banner
    legend_h = 56
    body_h = ch - TOP - legend_h
    half_w = cw // 2
    half_h = body_h // 2

    avail_w = half_w - 40
    avail_h = half_h - 100
    cell = min(avail_w // max(k, 1), avail_h // max(k, 1), 80)

    # --- TOP-LEFT: raw neighborhood ---
    draw_section_header(canvas, 0, TOP, half_w, 1, "PIXEL NEIGHBORHOOD",
                        f"The {k}x{k} pixels around ({x}, {y})")
    gx = (half_w - k * cell) // 2
    gy = TOP + 100
    draw_number_grid(canvas, patch, gx, gy, cell, cell,
                     text_color_fn=lambda v: C_TEXT)

    # --- TOP-RIGHT: SE mask (showing 1s in green, 0s in gray) ---
    draw_section_header(canvas, half_w, TOP, half_w, 2, "STRUCTURING ELEMENT",
                        f"Shape '{se_name}'", active=True)
    gx = half_w + (half_w - k * cell) // 2
    gy = TOP + 100
    draw_number_grid(canvas, se.astype(np.float32), gx, gy, cell, cell,
                     text_color_fn=lambda v: C_POS if v > 0 else C_ZERO,
                     border_color=C_AMBER)

    # --- BOTTOM-LEFT: active values only ---
    # Cells NOT under the SE are grayed out and stricken through.
    # Cells under the SE are highlighted and shown in green.
    draw_section_header(canvas, 0, TOP + half_h, half_w, 3, "ACTIVE VALUES",
                        "Only cells under the SE footprint")
    gx = (half_w - k * cell) // 2
    gy = TOP + half_h + 100
    draw_number_grid(canvas, patch, gx, gy, cell, cell,
                     text_color_fn=lambda v: C_POS if v > 0 else C_MUTED,
                     bg_fn=lambda r, c: C_PANEL_HI if se[r, c] > 0 else C_PANEL,
                     strike_fn=lambda r, c: se[r, c] == 0)

    # --- BOTTOM-RIGHT: the reduction (max or min) ---
    red = "max" if op == "dilate" else "min"
    verb = "dilation" if op == "dilate" else "erosion"
    draw_section_header(canvas, half_w, TOP + half_h, half_w, 4, "REDUCTION",
                        f"Reduce with {red} ({verb})")

    eq_x = half_w + 30
    eq_y = TOP + half_h + 130

    op_sym = "(f (+) b)" if op == "dilate" else "(f (-) b)"
    cv2.putText(canvas, f"{verb.capitalize()} {op_sym}",
                (eq_x, eq_y), F_TITLE, 0.85, C_AMBER, 2, cv2.LINE_AA)
    eq_y += 44

    # List the active values (up to 10, then "..." with total count)
    vals = [int(v) for v in active[:10]]
    vstr = ", ".join(str(v) for v in vals)
    if active.size > 10:
        vstr += f", ... ({active.size} total)"
    cv2.putText(canvas, f"= {red}{{ {vstr} }}",
                (eq_x, eq_y), F_BODY, 0.6, C_TEXT, 1, cv2.LINE_AA)
    eq_y += 60

    # The final reduction result
    cv2.putText(canvas, f"= {int(result)}",
                (eq_x, eq_y), F_TITLE, 1.4, C_POS, 3, cv2.LINE_AA)
    eq_y += 52
    cv2.putText(canvas, "-> output pixel",
                (eq_x, eq_y), F_BODY, 0.6, C_GREEN, 1, cv2.LINE_AA)

    # Dividers on top
    cv2.line(canvas, (half_w, TOP), (half_w, TOP + body_h), C_BORDER, 1)
    cv2.line(canvas, (0, TOP + half_h), (cw, TOP + half_h), C_BORDER, 1)

    draw_legend(canvas)
    return canvas


# ============================================================================
# SECTION 7 — PANEL 3: OTSU THRESHOLDING
# ============================================================================
# Otsu's method finds the intensity threshold T that maximally separates
# the image histogram into two classes ("dark" and "bright").
#
# The math:
#     sigma_B^2(T) = omega_0(T) * omega_1(T) * (mu_0(T) - mu_1(T))^2
#
# where:
#     omega_0(T) = fraction of pixels below T       (class 0 probability)
#     omega_1(T) = fraction of pixels above T       (class 1 probability)
#     mu_0(T)    = mean intensity of class 0
#     mu_1(T)    = mean intensity of class 1
#
# The optimal threshold:
#     T* = argmax_T sigma_B^2(T)
#
# This panel is drawn with matplotlib (not cv2) because we need proper
# plot axes for the histogram and the sigma_B^2 curve. It's rendered
# only every MATH_REDRAW_EVERY frames and cached.
# ----------------------------------------------------------------------------

def compute_otsu(gray):
    """
    Vectorized Otsu. Returns (T_star, sigma_B2_curve, hist, mu_T).

    The trick: instead of a Python loop over 256 candidate thresholds,
    we use cumulative sums (cumsum) to evaluate all of them at once.
    This turns O(256 * N) into O(256) for the histogram plus O(256)
    for the curve — instant on any CPU.
    """
    # 256-bin histogram of intensity values
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = int(hist.sum())
    if total == 0:
        return 0, np.zeros(256), hist, 0.0

    # Normalize to probabilities
    p = hist.astype(np.float64) / total

    # Cumulative sums for every candidate threshold T simultaneously
    omega = np.cumsum(p)                          # omega_0(T) for all T
    lv = np.arange(256, dtype=np.float64)         # intensity levels 0..255
    mu_cum = np.cumsum(lv * p)                    # sum_{i<=T} i * p_i
    mu_T = mu_cum[-1]                             # global mean

    # Between-class variance for all T at once. The denominator is
    # omega_0 * (1 - omega_0); we clamp it to avoid division by zero.
    denom = np.maximum(omega * (1.0 - omega), 1e-12)
    sigma_B2 = (mu_T * omega - mu_cum) ** 2 / denom

    # The optimal threshold is where sigma_B2 is largest
    return int(np.argmax(sigma_B2)), sigma_B2, hist, mu_T


def fig_to_bgr(fig):
    """
    Rasterize a matplotlib figure to a BGR numpy array, then close it.
    This is how we blit matplotlib output into a cv2 window.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR).copy()
    plt.close(fig)                                # free memory
    return bgr


def render_otsu_panel(gray, cw, ch):
    """
    Render the Otsu panel as a matplotlib figure at the given canvas
    size, then return it as a BGR array.

    GridSpec top=0.80 leaves room for the link banner drawn on top.
    """
    T_star, sigma_B2, hist, mu_T = compute_otsu(gray)

    # Create a figure sized to our canvas (converting px → inches at 100 dpi)
    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100,
                     facecolor="#121218")
    gs = GridSpec(1, 3, figure=fig, wspace=0.28,
                  left=0.05, right=0.98, top=0.80, bottom=0.14)

    # --- Left subplot: histogram ---
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1a1a24")
    ax1.bar(np.arange(256), hist, width=1.0, color="#4ea8de")
    ax1.axvline(T_star, color="#ff4757", linestyle="--", linewidth=2)
    ax1.set_title(f"Histogram  mu_T = {mu_T:.1f}", color="#e8e8f0", fontsize=14)
    ax1.tick_params(colors="#c8c8d8", labelsize=11)
    for s in ax1.spines.values(): s.set_color("#5a5a6e")

    # --- Middle subplot: sigma_B^2(T) curve ---
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#1a1a24")
    ax2.plot(sigma_B2, color="#7dffa0", linewidth=3)
    ax2.axvline(T_star, color="#ff4757", linestyle="--", linewidth=2)
    ax2.plot(T_star, sigma_B2[T_star], "o", color="#ff4757", markersize=12)
    ax2.set_title("sigma_B^2(T)", color="#e8e8f0", fontsize=14)
    ax2.tick_params(colors="#c8c8d8", labelsize=11)
    for s in ax2.spines.values(): s.set_color("#5a5a6e")

    # --- Right subplot: arithmetic card ---
    # Recompute the class statistics at T* to print them on screen.
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#1a1a24")
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values(): s.set_color("#5a5a6e")

    p = hist.astype(np.float64) / max(hist.sum(), 1)
    o0 = p[: T_star + 1].sum()                    # omega_0
    o1 = 1.0 - o0                                 # omega_1
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
# SECTION 8 — PANEL 4: HOUGH TRANSFORM
# ============================================================================
# The Hough transform detects lines by voting in (ρ, θ) parameter space.
#
# For every edge pixel (x, y), it votes for every possible line through
# that point. A line is parameterized in NORMAL form:
#
#     ρ = x * cos(θ) + y * sin(θ)
#
# where ρ is the perpendicular distance from the origin, and θ is the
# angle of the normal to the line. In (ρ, θ) space, a single edge pixel
# traces a sinusoid. Where many sinusoids cross, many pixels agree on a
# line — that crossing is a peak in the accumulator A(ρ, θ).
#
# This panel shows:
#   - Left: the edge map (Canny output) with detected lines overlaid
#   - Right: the accumulator A(ρ, θ) as a heatmap with peaks marked
# ----------------------------------------------------------------------------

def hough_lines(edges, n=3, min_votes=20):
    """
    Compute a Hough accumulator and return the top-N peaks.

    Returns
    -------
    acc    : the accumulator array (n_rho, n_theta)
    rhos   : array of ρ values corresponding to acc rows
    thetas : array of θ values corresponding to acc columns
    peaks  : list of (rho, theta, votes) for the top N peaks
    """
    h, w = edges.shape

    # ρ ranges from -diag to +diag, where diag is the image diagonal
    diag = int(np.ceil(np.hypot(h, w)))
    rhos = np.arange(-diag, diag + 1, 1.0)
    thetas = np.arange(0, np.pi, np.pi / 180.0)     # 1° resolution

    # Accumulator: rows = ρ, columns = θ
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    # Get all edge pixel coordinates
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return acc, rhos, thetas, []

    # Subsample if there are too many edge pixels (performance cap)
    if len(xs) > 1500:
        idx = np.random.choice(len(xs), 1500, replace=False)
        xs, ys = xs[idx], ys[idx]

    # Vectorized voting: for each edge pixel, compute ρ at every θ
    ct, st = np.cos(thetas), np.sin(thetas)
    for x, y in zip(xs, ys):
        r = x * ct + y * st                             # ρ for all θ
        ri = np.round(r + diag).astype(int)             # shift to 0-based
        ok = (ri >= 0) & (ri < len(rhos))               # filter valid bins
        acc[ri[ok], np.arange(len(thetas))[ok]] += 1    # vote

    # Greedy peak extraction with local suppression
    # (so we don't re-detect the same line multiple times)
    aw = acc.copy()
    peaks = []
    for _ in range(n):
        if aw.max() < min_votes:
            break
        ri, ti = np.unravel_index(np.argmax(aw), aw.shape)
        peaks.append((float(rhos[ri]), float(thetas[ti]), int(aw[ri, ti])))
        # Zero out a neighborhood around the peak (±20 rows, ±10 cols)
        aw[max(0, ri - 20):ri + 21, max(0, ti - 10):ti + 11] = 0

    return acc, rhos, thetas, peaks


def render_hough_panel(edges, cw, ch):
    """
    Render the Hough panel as a matplotlib figure, return it as BGR.

    GridSpec top=0.80 leaves room for the link banner drawn on top.
    """
    acc, rhos, thetas, peaks = hough_lines(edges)

    fig = plt.figure(figsize=(cw / 100, ch / 100), dpi=100,
                     facecolor="#121218")
    gs = GridSpec(1, 2, figure=fig, wspace=0.22,
                  left=0.05, right=0.98, top=0.80, bottom=0.14)

    # --- Left subplot: edge map with detected lines overlaid ---
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#1a1a24")
    ax1.imshow(edges, cmap="gray")
    ax1.set_title("Edge map + detected lines", color="#e8e8f0", fontsize=14)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Draw the detected lines back on the edge map
    h, w = edges.shape
    for (rho, theta, _) in peaks:
        c, s = np.cos(theta), np.sin(theta)
        # Two points on the line: x=0 and x=w-1, solve for y
        ax1.plot([0, w - 1], [rho / max(s, 1e-9),
                              (rho - (w - 1) * c) / max(s, 1e-9)],
                 "-", color="#ff4757", linewidth=2)

    # --- Right subplot: accumulator ---
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#1a1a24")
    td = np.degrees(thetas)
    ax2.imshow(acc, cmap="inferno", aspect="auto",
               extent=[td[0], td[-1], rhos[0], rhos[-1]], origin="lower")
    ax2.set_xlabel("theta (deg)", color="#c8c8d8", fontsize=11)
    ax2.set_ylabel("rho (px)", color="#c8c8d8", fontsize=11)
    ax2.set_title("Accumulator A(rho, theta)", color="#e8e8f0", fontsize=14)
    ax2.tick_params(colors="#c8c8d8", labelsize=11)
    for s in ax2.spines.values(): s.set_color("#5a5a6e")

    # Mark each detected peak with a green circle + label
    for i, (rho, theta, v) in enumerate(peaks):
        ax2.plot(np.degrees(theta), rho, "o", markerfacecolor="none",
                 markeredgecolor="#7dffa0", markeredgewidth=2.5,
                 markersize=14)
        ax2.text(np.degrees(theta) + 4, rho, f"  #{i+1}: {v}",
                 color="#7dffa0", fontsize=10, fontweight="bold")

    return fig_to_bgr(fig)


# ============================================================================
# SECTION 9 — FACE DETECTION
# ============================================================================
# Two backends, tried in order:
#   1. OpenCV DNN with a Caffe SSD model — more accurate, needs 2 model
#      files (deploy.prototxt + res10_300x300_ssd_iter_140000_fp16.caffemodel)
#   2. Haar cascade classifier — ships with OpenCV, no external files
#
# The DNN returns the LARGEST high-confidence face box, not the highest-
# confidence one. Reason: on a webcam, the biggest face is usually the
# user; a small high-confidence face could be a poster in the background.
# ----------------------------------------------------------------------------

def load_face_detector(script_dir):
    """
    Try DNN first, fall back to Haar. Returns (kind, detector).
    kind is one of: "dnn", "haar", "none".
    """
    proto = os.path.join(script_dir, "deploy.prototxt")
    model = os.path.join(script_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    if os.path.exists(proto) and os.path.exists(model):
        try:
            net = cv2.dnn.readNetFromCaffe(proto, model)
            print(f"[focusguard] DNN face detector loaded")
            return "dnn", net
        except cv2.error as e:
            print(f"[focusguard] DNN failed: {e}")

    # Haar fallback — always available because OpenCV ships the XML
    haar = os.path.join(cv2.data.haarcascades,
                        "haarcascade_frontalface_default.xml")
    if os.path.exists(haar):
        print("[focusguard] Using Haar cascade fallback")
        return "haar", cv2.CascadeClassifier(haar)

    return "none", None


def detect_face_dnn(net, frame_bgr, conf=0.6):
    """
    Run the DNN, return the LARGEST high-confidence face box (x1,y1,x2,y2)
    in the coordinates of the input frame. Returns None if no face found.
    """
    h, w = frame_bgr.shape[:2]

    # The DNN expects a 300×300 blob with mean-subtracted channels.
    # The (104, 177, 123) is the ImageNet BGR mean.
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_bgr, (300, 300)),
                                 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    det = net.forward()

    # det shape is (1, 1, N, 7) where each row is
    # [image_id, class_id, confidence, x1, y1, x2, y2] (normalized 0..1)
    best, best_area = None, 0
    for i in range(det.shape[2]):
        c = float(det[0, 0, i, 2])
        if c < conf:
            continue
        # Scale the normalized box back to pixel coordinates
        b = det[0, 0, i, 3:7] * np.array([w, h, w, h])
        area = (b[2] - b[0]) * (b[3] - b[1])
        # Filter out tiny boxes (likely false positives)
        if area < 0.02 * w * h:
            continue
        # Keep the largest
        if area > best_area:
            best = b.astype(int)
            best_area = area
    return best


def detect_face_haar(haar, gray_small):
    """
    Haar cascade fallback. Returns the LARGEST + most central face box,
    or None if no face detected.
    """
    boxes = haar.detectMultiScale(gray_small, scaleFactor=1.15,
                                  minNeighbors=6, minSize=(80, 80))
    if len(boxes) == 0:
        return None

    # Score each box: prefer large AND centered (avoids picking up
    # background faces from posters or people walking by)
    h, w = gray_small.shape[:2]
    cx, cy = w / 2, h / 2
    best, best_score = None, -1
    for (x, y, bw, bh) in boxes:
        bx, by = x + bw / 2, y + bh / 2
        d = np.hypot(bx - cx, by - cy)          # distance from frame center
        score = (bw * bh) - d * 50              # large = good, far = bad
        if score > best_score:
            best = np.array([x, y, x + bw, y + bh], dtype=int)
            best_score = score
    return best


# ============================================================================
# SECTION 10 — MOOD PIPELINE (AU DETECTORS)
# ============================================================================
# Each function takes a small ROI (region of interest) from the face and
# returns a value in [0, 1] representing the activation strength of one
# or two Facial Action Units.
#
# These are PROXIES, not clinically validated AU detectors. But they use
# exactly the classical CV primitives from the lecture slides: Otsu,
# morphological opening/closing, contour fitting, adaptive threshold,
# Canny, Hough.
# ----------------------------------------------------------------------------

def mouth_au(gray, roi):
    """
    Mouth ROI → Otsu threshold → morphological opening + closing →
    largest contour → fit parabola → curvature is AU12; aspect ratio is AU25.

    Returns (au12, au25, mask). Returns zeros if no valid mouth shape
    is found — this is important because a hand on the chin could
    otherwise be misclassified as a big smile.
    """
    if roi.size < 100:
        return 0.0, 0.0, np.zeros_like(gray)

    # --- STEP 1: Otsu threshold ---
    # Lips are darker than the surrounding skin, so INV makes the lip
    # region white in the binary mask.
    T, _, _, _ = compute_otsu(roi)
    _, mask = cv2.threshold(roi, T, 255, cv2.THRESH_BINARY_INV)

    # --- STEP 2: Morphological opening + closing ---
    # Opening (erode → dilate) removes lip speckle / salt noise.
    # Closing (dilate → erode) fills small gaps between lips and teeth.
    # The 5×5 ellipse SE is rotationally invariant (no orientation bias).
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)     # A ∘ B
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)    # A • B

    # --- STEP 3: Largest contour ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.0, mask
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    roi_area = float(roi.shape[0] * roi.shape[1])

    # --- STEP 4: Area validation (reject hands, chins, noise) ---
    # A real mouth region occupies 3%-55% of the mouth ROI.
    # Below 3% is noise; above 55% is almost certainly a hand or chin.
    if area < 0.03 * roi_area or area > 0.55 * roi_area:
        return 0.0, 0.0, mask
    if len(c) < 5:
        return 0.0, 0.0, mask

    # --- STEP 5: Fit a 2nd-order parabola y = a·x² + b·x + c ---
    # The sign and magnitude of `a` tells us the curvature of the upper
    # lip line. For a smile, the corners lift (a < 0 in image coords
    # because y grows downward), so -a is positive.
    pts = c.reshape(-1, 2)
    try:
        a = float(np.polyfit(pts[:, 0].astype(np.float64),
                             pts[:, 1].astype(np.float64), 2)[0])
    except Exception:
        a = 0.0

    # --- STEP 6: Aspect ratio validation ---
    # A mouth is wider than tall. If the contour is tall, it's probably
    # an open mouth or noise — reject to avoid false "talking" signals.
    _, _, bw, bh = cv2.boundingRect(c)
    if bh > 0.5 * bw:
        return 0.0, 0.0, mask

    # AU12: smile curvature. -a * 200 scaled empirically.
    au12 = float(np.clip(-a * 200.0, 0.0, 1.0))
    # AU25: lips part (openness). bh/bw normalized by 0.6.
    au25 = float(np.clip(bh / max(bw, 1) / 0.6, 0.0, 1.0))
    return au12, au25, mask


def eye_au(gray, roi):
    """
    Eye ROI → adaptive threshold → dark pixel ratio → AU6 (cheek raise).

    Adaptive threshold handles changing classroom lighting better than
    global threshold: it computes a local threshold per pixel from a
    15×15 neighborhood. This is the same concept as Otsu but local.

    The dark pixel ratio correlates with eye openness: wide-open eyes
    have more pupil visible (higher ratio), squinted eyes have less.
    AU6 is high when the eye is squinting (cheeks raised, Duchenne smile).
    """
    if roi.size < 100:
        return 0.0, np.zeros_like(gray)

    mask = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 5)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)

    # Fraction of dark (below-threshold) pixels
    dr = float((mask > 0).sum()) / max(mask.size, 1)

    # AU6 proxy: inverted openness, scaled so a very open eye → ~0
    # and a squinted eye → ~1. The 0.25 is an empirical "fully open" cap.
    return float(np.clip(1.0 - dr / 0.25, 0.0, 1.0)), mask


def brow_au(gray, roi):
    """
    Brow ROI → Canny edges → Hough lines → mean signed angle.

    A brow that slants up toward the center = AU1 (inner brow raiser).
    A brow that slants down toward the center = AU4 (brow lowerer).

    The sign of the mean line angle discriminates between them.
    Returns (au1, au4, edges).
    """
    if roi.size < 100:
        return 0.0, 0.0, np.zeros_like(gray)

    # Canny edge detection (see Canny slide: Gaussian → gradient → NMS → hysteresis)
    edges = cv2.Canny(roi, 50, 150)

    # Hough for line segments (see Hough slide)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                            minLineLength=15, maxLineGap=8)
    if lines is None or len(lines) == 0:
        return 0.0, 0.0, edges

    # Compute the signed angle of each line (degrees)
    sgn = [np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
           for l in lines]
    m = float(np.mean(sgn))

    # Normalize to [0, 1] with a ±20° range
    au1 = float(np.clip(m / 20.0, 0.0, 1.0))     # brow up (positive angle)
    au4 = float(np.clip(-m / 20.0, 0.0, 1.0))    # brow down (negative angle)
    return au1, au4, edges


def init_kalman(n=5):
    """
    Initialize a 5-state Kalman filter for temporal AU smoothing.

    State vector = [AU12, AU6, AU4, AU1, AU25]
    Transition = identity (each state evolves independently)
    Measurement = identity (we observe each state directly)

    Process noise: how much we expect the true AU to change frame to frame.
    Measurement noise: how noisy the raw AU measurement is.

    The filter produces a smooth estimate that doesn't jitter on noise
    but still tracks real expression changes.
    """
    kf = cv2.KalmanFilter(n, n)
    kf.transitionMatrix = np.eye(n, dtype=np.float32)
    kf.measurementMatrix = np.eye(n, dtype=np.float32)
    kf.processNoiseCov = np.eye(n, dtype=np.float32) * 1e-3      # low process noise
    kf.measurementNoiseCov = np.eye(n, dtype=np.float32) * 1e-1  # higher measurement noise
    kf.errorCovPost = np.eye(n, dtype=np.float32)
    kf.statePost = np.zeros((n, 1), dtype=np.float32)
    return kf


def classify_mood(au):
    """
    Rule-based mood classifier on the smoothed AU vector.

    Rules are ordered by priority: the first matching rule wins.
    These thresholds were tuned by trial on typical webcam footage.
    """
    a12, a6, a4, a1, a25 = au
    if a12 > 0.40 and a6 > 0.15:  return "happy"       # smile + cheek raise
    if a4 > 0.40 and a1 < 0.20:   return "focused"     # brow furrow
    if a1 > 0.35 and a25 > 0.25:  return "surprised"   # brow raise + mouth open
    if a25 > 0.40:                return "talking"     # mouth open
    return "neutral"


# ============================================================================
# SECTION 11 — LIVENESS DETECTION
# ============================================================================
# Optical flow on face corners. A real face has micro-motion (breathing,
# blinking, head drift). A printed photo taped to a stick doesn't.
#
# We use Lucas-Kanade sparse optical flow: pick 30 good corners in the
# face box, track them between consecutive frames, measure the mean
# displacement magnitude. If it's below a threshold, flag as suspicious.
# ----------------------------------------------------------------------------

# Lucas-Kanade parameters for cv2.calcOpticalFlowPyrLK
LK = dict(
    winSize=(15, 15),                    # neighborhood size
    maxLevel=2,                          # pyramid levels for multi-scale
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)


def liveness_flow(gray, box):
    """
    Return mean optical-flow magnitude in the face box (px/frame).

    State (prev_gray, prev_pts) is kept in STATE and updated each call,
    so this function has memory across frames.
    """
    if box is None:
        STATE["prev_gray"] = gray
        STATE["prev_pts"] = None
        return 0.0

    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = gray.shape
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return 0.0

    # Mask so goodFeaturesToTrack only picks corners inside the face box
    m = np.zeros_like(gray)
    m[y1:y2, x1:x2] = 255

    pg, pp = STATE["prev_gray"], STATE["prev_pts"]

    # First frame (or lost points): just detect corners and store them
    if pg is None or pp is None or len(pp) < 5:
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=30,
                                      qualityLevel=0.01, minDistance=10,
                                      mask=m)
        STATE["prev_gray"] = gray
        STATE["prev_pts"] = pts
        return 0.0

    # Track the corners from the previous frame to this frame
    np_, st, _ = cv2.calcOpticalFlowPyrLK(pg, gray, pp, None, **LK)
    STATE["prev_gray"] = gray

    if np_ is None or st is None:
        STATE["prev_pts"] = None
        return 0.0

    # Filter to points that were successfully tracked
    good = st.ravel() == 1
    if good.sum() < 3:
        STATE["prev_pts"] = np_
        return 0.0

    # Mean displacement magnitude of the good tracks
    f = float(np.linalg.norm(np_[good] - pp[good], axis=2).mean())
    STATE["prev_pts"] = np_
    return f


# ============================================================================
# SECTION 12 — FACE ANALYSIS ORCHESTRATOR
# ============================================================================
# This is the "conductor" that takes a face box, crops the 3 ROIs,
# calls the AU detectors, applies the Kalman filter, runs the liveness
# check, and computes the focus score.
#
# Returns a dict ready to hand to draw_overlay(). Nothing in here touches
# the display — that's the overlay's job.
# ----------------------------------------------------------------------------

def analyze_face(gray_small):
    """Run the full mood/focus pipeline for one frame."""
    box = STATE["face_box"]

    # Default result — returned unchanged if no face found
    res = {"face_box": box, "au": np.zeros(5, dtype=np.float32),
           "mood": "neutral", "focus": 0.0, "motion": 0.0}
    if box is None:
        return res

    h, w = gray_small.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    # Clamp the box to image bounds
    x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
    fh, fw = y2 - y1, x2 - x1
    if fh < 60 or fw < 60:
        return res                              # too small to analyze

    # --- Define the three ROI bands as fractions of face height ---
    # These fractions were tuned empirically:
    #   brow:  15%-35% down from top of face
    #   eye:   35%-55% down
    #   mouth: 70%-95% down, and 25%-75% across (center-ish)
    brow_y0, brow_y1 = y1 + int(fh * 0.15), y1 + int(fh * 0.35)
    eye_y0, eye_y1 = y1 + int(fh * 0.35), y1 + int(fh * 0.55)
    mouth_y0, mouth_y1 = y1 + int(fh * 0.70), y1 + int(fh * 0.95)
    mouth_x0, mouth_x1 = x1 + int(fw * 0.25), x1 + int(fw * 0.75)

    # Crop the ROIs
    brow_roi = gray_small[brow_y0:brow_y1, x1:x2]
    eye_roi = gray_small[eye_y0:eye_y1, x1:x2]
    mouth_roi = gray_small[mouth_y0:mouth_y1, mouth_x0:mouth_x1]

    # --- Run the AU detectors ---
    au12, au25, _ = mouth_au(gray_small, mouth_roi)
    au6, _ = eye_au(gray_small, eye_roi)
    au1, au4, _ = brow_au(gray_small, brow_roi)

    # --- Assemble the raw 5-D AU vector ---
    # ORDER IS CRITICAL: must match AU_INFO and classify_mood
    raw = np.array([au12, au6, au4, au1, au25], dtype=np.float32)

    # --- Kalman smoothing ---
    if STATE["mood_kf"] is None:
        STATE["mood_kf"] = init_kalman(5)
    STATE["mood_kf"].correct(raw.reshape(-1, 1).astype(np.float32))
    smooth = np.clip(STATE["mood_kf"].predict().ravel(), 0.0, 1.0)

    # --- Liveness ---
    motion = liveness_flow(gray_small, box) if STATE["show_liveness"] else 0.0

    # --- Blink detection from the RAW eye signal ---
    # We use the raw (unsmoothed) AU6 because blinks are transient spikes
    # that the Kalman filter would suppress. 0.85 = eye almost fully closed.
    now = time.time()
    if float(raw[1]) > 0.85:
        last = STATE["blink_times"][-1] if STATE["blink_times"] else 0
        if now - last > 0.20:                   # 200ms debounce
            STATE["blink_times"].append(now)

    # Blinks per minute over the last 60 seconds
    br = sum(1 for t in STATE["blink_times"] if now - t < 60.0)

    # --- Focus score ---
    # Start from 100, subtract for blinks (capped at 40) and motion
    # (weighted 60x). Clip to [0, 100].
    focus = float(np.clip(100.0 - min(1.5 * br, 40.0) - 60.0 * motion,
                          0.0, 100.0))

    res.update({"au": smooth, "mood": classify_mood(smooth),
                "focus": focus, "motion": motion})
    return res


# ============================================================================
# SECTION 13 — OVERLAY (FocusGuard window)
# ============================================================================
# Draws everything on top of the video frame:
#   - Top header strip: "FocusGuard" + mood + link to math panel
#   - Left accent strip (color shared with math panel)
#   - Face detection box
#   - FACIAL SIGNALS panel (compact AU dashboard)
#   - FOCUS SCORE card
#   - Bottom strip: "Why this mood?" + liveness indicator
#   - Yellow cursor crosshair
# ----------------------------------------------------------------------------

# Mood → color mapping (used for the mood label in the header)
MOOD_COLORS = {"happy": C_POS, "focused": C_BLUE, "surprised": C_AMBER,
               "talking": C_PURPLE, "neutral": C_MUTED}

# Mood → plain-English explanation (shown at the bottom)
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
    """
    Draw all overlays on top of the FocusGuard canvas.

    canvas     : the BGR canvas (already has the video frame blitted in)
    data       : dict from analyze_face() — face box, AUs, mood, focus
    frame_rect : (x, y, w, h) of where the video sits inside the canvas
    mouse_xy   : (x, y) of the cursor in canvas coordinates
    """
    H, W = canvas.shape[:2]
    fx, fy, fw, fh = frame_rect
    accent = PANEL_ACCENT[STATE["panel"]]

    # ---------- HEADER STRIP (56px tall) ----------
    # Semi-transparent dark strip so text stays readable over the video.
    hh = 56
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (W, hh), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)

    # App name (left)
    cv2.putText(canvas, "FocusGuard", (22, 30), F_TITLE, 0.85,
                C_TEXT, 2, cv2.LINE_AA)

    # Mood (right, color-coded)
    mood_lbl = f"Mood: {data['mood']}"
    mc = MOOD_COLORS.get(data["mood"], C_TEXT)
    (tw, _), _ = cv2.getTextSize(mood_lbl, F_TITLE, 0.85, 2)
    cv2.putText(canvas, mood_lbl, (W - tw - 22, 30), F_TITLE, 0.85,
                mc, 2, cv2.LINE_AA)

    # Link to math panel (sub-line, in the accent color)
    link = f"-> Math Panel [{PANEL_KEY[STATE['panel']]}]: " \
           f"{PANEL_NAMES[STATE['panel']]}"
    cv2.putText(canvas, link, (22, 50), F_BODY, 0.46, accent, 1, cv2.LINE_AA)

    # Left accent strip (matches the math panel's left edge)
    cv2.rectangle(canvas, (0, 0), (6, H), accent, -1)

    # ---------- FACE DETECTION BOX ----------
    # face_box is stored in 480×360 (small) coords; scale UP to canvas.
    if data["face_box"] is not None:
        bx1, by1, bx2, by2 = [int(v) for v in data["face_box"]]
        sx, sy = fw / WORK_W, fh / WORK_H         # small → canvas scale
        cv2.rectangle(canvas,
                      (fx + int(bx1 * sx), fy + int(by1 * sy)),
                      (fx + int(bx2 * sx), fy + int(by2 * sy)),
                      C_GREEN, 3, cv2.LINE_AA)

    # ---------- FACIAL SIGNALS PANEL (compact, top-left) ----------
    px, py = 16, hh + 12
    pw, ph = 340, 178

    # Panel background (90% opacity for readability over bright scenes)
    ov = canvas.copy()
    cv2.rectangle(ov, (px, py), (px + pw, py + ph), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.90, canvas, 0.10, 0, canvas)
    cv2.rectangle(canvas, (px, py), (px + pw, py + ph),
                  C_BORDER, 2, cv2.LINE_AA)

    # Panel title and subtitle
    cv2.putText(canvas, "FACIAL SIGNALS", (px + 14, py + 22),
                F_TITLE, 0.55, C_AMBER, 2, cv2.LINE_AA)
    cv2.putText(canvas, "5 signals, range 0.00 - 1.00",
                (px + 14, py + 40), F_BODY, 0.36, C_MUTED, 1, cv2.LINE_AA)
    cv2.line(canvas, (px + 14, py + 48),
             (px + pw - 14, py + 48), C_BORDER, 1)

    # Column x-positions for the row layout:
    #   ●  <friendly name>  <code>  [bar]  <state>  <value>
    row_y0 = py + 58
    row_h = 24
    dot_x = px + 14
    name_x = px + 26
    tech_x = px + 122
    bar_x = px + 168
    bar_w = 70
    state_x = px + 244
    val_x = px + 296

    for i, (short, tech) in enumerate(AU_INFO):
        v = float(data["au"][i])
        state, scol = _au_state(v)
        y = row_y0 + i * row_h

        # Colored dot indicating the state (green/amber/gray)
        cv2.circle(canvas, (dot_x, y + 8), 4, scol, -1, cv2.LINE_AA)

        # Friendly name + technical code
        cv2.putText(canvas, short, (name_x, y + 12),
                    F_TITLE, 0.42, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, tech, (tech_x, y + 12),
                    F_BODY, 0.32, C_MUTED, 1, cv2.LINE_AA)

        # Progress bar: background + fill (color = state color)
        cv2.rectangle(canvas, (bar_x, y + 4),
                      (bar_x + bar_w, y + 12), (40, 40, 52), -1)
        fill = int(bar_w * np.clip(v, 0, 1))
        cv2.rectangle(canvas, (bar_x, y + 4),
                      (bar_x + fill, y + 12), scol, -1)

        # HIGH/MED/LOW label and numeric value
        cv2.putText(canvas, state, (state_x, y + 12),
                    F_BODY, 0.34, scol, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{v:.2f}", (val_x, y + 12),
                    F_BODY, 0.32, C_MUTED, 1, cv2.LINE_AA)

    # ---------- FOCUS SCORE CARD (top-right) ----------
    card_w, card_h = 220, ph                    # same height as AU panel
    cx = W - card_w - 16
    cy = py

    ov = canvas.copy()
    cv2.rectangle(ov, (cx, cy), (cx + card_w, cy + card_h), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.90, canvas, 0.10, 0, canvas)
    cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h),
                  C_BORDER, 2, cv2.LINE_AA)

    cv2.putText(canvas, "FOCUS SCORE", (cx + 14, cy + 22),
                F_TITLE, 0.55, C_AMBER, 2, cv2.LINE_AA)
    cv2.putText(canvas, "how engaged you look", (cx + 14, cy + 40),
                F_BODY, 0.36, C_MUTED, 1, cv2.LINE_AA)

    # Big number, color-coded by value
    fs = int(data["focus"])
    fcol = C_GREEN if fs > 65 else (C_AMBER if fs > 35 else C_RED)
    big = str(fs)
    (bw, bh), _ = cv2.getTextSize(big, F_TITLE, 1.8, 3)
    cv2.putText(canvas, big, (cx + card_w - bw - 14, cy + 110),
                F_TITLE, 1.8, fcol, 3, cv2.LINE_AA)

    # Focus bar (0 → 100)
    bar_x2 = cx + 14
    bar_y2 = cy + 130
    bar_w2 = card_w - 28
    cv2.rectangle(canvas, (bar_x2, bar_y2),
                  (bar_x2 + bar_w2, bar_y2 + 10), (40, 40, 52), -1)
    cv2.rectangle(canvas, (bar_x2, bar_y2),
                  (bar_x2 + int(bar_w2 * fs / 100), bar_y2 + 10), fcol, -1)
    cv2.putText(canvas, "0", (bar_x2, bar_y2 + 26),
                F_BODY, 0.32, C_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "100", (bar_x2 + bar_w2 - 22, bar_y2 + 26),
                F_BODY, 0.32, C_MUTED, 1, cv2.LINE_AA)

    # ---------- BOTTOM STRIP: mood explanation + liveness ----------
    strip_h = 48
    sy = H - strip_h
    ov = canvas.copy()
    cv2.rectangle(ov, (0, sy), (W, H), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
    cv2.line(canvas, (0, sy), (W, sy), accent, 2)

    why = MOOD_MEANING.get(data["mood"], "")
    cv2.putText(canvas, f"Why this mood?  '{data['mood']}' = {why}",
                (20, sy + 30), F_BODY, 0.52, C_TEXT, 1, cv2.LINE_AA)

    # Liveness indicator (right side of bottom strip)
    if STATE["show_liveness"]:
        live = "LIVE PERSON" if data["motion"] > 0.15 else "STILL (spoof?)"
        lc = C_GREEN if data["motion"] > 0.15 else C_RED
        (lw, _), _ = cv2.getTextSize(live, F_TITLE, 0.5, 2)
        cv2.putText(canvas, live, (W - lw - 20, sy + 30),
                    F_TITLE, 0.5, lc, 2, cv2.LINE_AA)

    # ---------- CURSOR CROSSHAIR ----------
    # Drawn last so it's always on top of everything else
    mx, my = mouse_xy
    if 0 <= mx < W and 0 <= my < H:
        cv2.drawMarker(canvas, (mx, my), C_YELLOW,
                       cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)


# ============================================================================
# SECTION 14 — WINDOW AND LAYOUT SETUP
# ============================================================================
# Two concerns here:
#   1. Query the screen size (using tkinter) so we can size windows
#      proportionally on any display.
#   2. Compute a side-by-side layout with sensible margins and a
#      taskbar reserve at the bottom.
#
# Fallback: if tkinter is unavailable (headless), use 1920×1080.
# ----------------------------------------------------------------------------

def get_screen():
    """Return (screen_width, screen_height). Falls back to 1920×1080."""
    try:
        import tkinter as tk
        r = tk.Tk(); r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def compute_layout(sw, sh):
    """
    Compute window geometry for both windows.

    Horizontal layout (screen ≥ 1200px wide):
      FocusGuard 40% of available width, Math Panel 60%.

    Vertical layout (screen < 1200px wide):
      FocusGuard on top (45% of height), Math Panel below.

    Margins: 20px on all sides, 20px gutter between windows,
    60px reserved at bottom for the taskbar.
    """
    M, G, T, B = 20, 20, 20, 60     # margin, gutter, top, bottom
    aw, ah = sw - 2 * M - G, sh - T - B   # available width, height

    if sw < 1200:                   # vertical stack
        mw = aw + G                 # full width
        a_w = mw
        mh = int(ah * 0.45)
        a_h = ah - mh - G
        return {"vertical": True, "margin": M, "gutter": G, "top": T,
                "main": (M, T, mw, mh),
                "math": (M, T + mh + G, a_w, a_h)}

    # Horizontal side-by-side
    mw = int(aw * 0.42)             # FocusGuard 42%
    a_w = aw - mw                   # Math Panel = rest
    mh = ah
    a_h = ah
    return {"vertical": False, "margin": M, "gutter": G, "top": T,
            "main": (M, T, mw, mh),
            "math": (M + mw + G, T, a_w, a_h)}


def fit_frame(frame, rw, rh):
    """
    Resize a frame to fit inside (rw, rh) while preserving aspect ratio.
    Returns (resized, x_offset, y_offset, new_w, new_h).

    The offsets are used to center the fitted frame inside the target
    rectangle (letterboxing).
    """
    fh, fw = frame.shape[:2]
    s = min(rw / fw, rh / fh)
    nw, nh = int(fw * s), int(fh * s)
    return (cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR),
            (rw - nw) // 2, (rh - nh) // 2, nw, nh)


# ============================================================================
# SECTION 15 — MATH PANEL DISPATCH (caching layer)
# ============================================================================
# The math panel is expensive to draw (especially Otsu/Hough with
# matplotlib). Two caching strategies:
#
#   1. Kernel/Morph panels: keyed by (panel, indices, mouse_x, mouse_y).
#      If none of these changed, reuse the cached canvas. Redrawn only
#      when the user moves the mouse to a new pixel or changes panel.
#
#   2. Otsu/Hough panels: redrawn every MATH_REDRAW_EVERY frames
#      (they depend on the whole image, not on the mouse).
# ----------------------------------------------------------------------------

def math_signature():
    """
    A tuple of everything the kernel/morph panels depend on. If this
    tuple is unchanged, the cached canvas is still valid.
    """
    return (STATE["panel"], STATE["kernel_idx"], STATE["se_idx"],
            STATE["op_idx"], STATE["mouse_small_x"], STATE["mouse_small_y"])


def decorate_math_panel(canvas):
    """
    Draw the shared decorations on a math panel canvas:
      - Left accent strip (matches FocusGuard's left strip)
      - Top banner showing "LINKED TO FOCUSGUARD ..." and the panel name
    """
    h, w = canvas.shape[:2]
    accent = PANEL_ACCENT[STATE["panel"]]

    # Left accent strip (same color/width as FocusGuard's left strip)
    cv2.rectangle(canvas, (0, 0), (7, h), accent, -1)

    # Semi-transparent banner across the top
    banner_h = 30
    ov = canvas.copy()
    cv2.rectangle(ov, (7, 0), (w, banner_h), (10, 10, 14), -1)
    cv2.addWeighted(ov, 0.90, canvas, 0.10, 0, canvas)
    cv2.line(canvas, (7, banner_h), (w, banner_h), accent, 2)

    # The banner text tells the viewer which key keeps this panel active
    label = f"<- LINKED TO FOCUSGUARD  |  {PANEL_NAMES[STATE['panel']]}  " \
            f"|  press [{PANEL_KEY[STATE['panel']]}] on FocusGuard to keep this view"
    cv2.putText(canvas, label, (22, 20), F_BODY, 0.55, accent, 2,
                cv2.LINE_AA)

    return canvas


def get_math_panel(gray_small, fc, cw, ch):
    """
    Return the current math panel canvas, using caches to avoid
    re-rendering when nothing has changed.
    """
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

    # Draw the shared decorations on top
    c = decorate_math_panel(c)

    # Cache for next time
    STATE["math_cache"] = c
    STATE["math_sig"] = sig
    return c


# ============================================================================
# SECTION 16 — INTERACTION (mouse + keyboard callbacks)
# ============================================================================

def on_mouse(event, x, y, flags, _):
    """Track mouse position over the FocusGuard window."""
    if event == cv2.EVENT_MOUSEMOVE:
        STATE["mouse"] = (x, y)


def handle_key(k):
    """
    Handle a keypress. Return True to keep running, False to quit.
    Returns True for key=-1 (no key pressed) so the caller can loop.
    """
    if k in (255, -1):              # no key
        return True
    if k == ord("q"):
        return False                # quit signal
    if k == ord("1"):   STATE["panel"] = "kernel"
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

    # Invalidate the math panel cache — anything could have changed
    STATE["math_sig"] = None
    return True


# ============================================================================
# SECTION 17 — MAIN LOOP
# ============================================================================
# Flow per frame:
#   1. Read frame, mirror it (so the user sees themselves naturally)
#   2. Downscale to 480×360 for CV work
#   3. Run face detection every N frames
#   4. Run the mood pipeline (analyze_face)
#   5. Render FocusGuard canvas (video + overlay)
#   6. Render Math Panel canvas (4-panel dispatch)
#   7. Show both windows
#   8. Handle keyboard
#   9. Check for window-close events
#
# try/finally ensures the camera is released and windows are destroyed
# even on exceptions or Ctrl+C.
# ----------------------------------------------------------------------------

def main():
    # Locate script directory for optional DNN model files
    here = os.path.dirname(os.path.abspath(__file__))
    kind, detector = load_face_detector(here)

    # Open the webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[focusguard] cannot open camera {CAM_INDEX}")
        return

    # Query screen and compute geometry
    sw, sh = get_screen()
    L = compute_layout(sw, sh)
    mx_, my_, mw_, mh_ = L["main"]
    ax_, ay_, aw_, ah_ = L["math"]
    print(f"[focusguard] Screen: {sw}x{sh} "
          f"({'vertical' if L['vertical'] else 'horizontal'})")
    print(f"[focusguard] FocusGuard: {mw_}x{mh_} at ({mx_}, {my_})")
    print(f"[focusguard] Math Panel: {aw_}x{ah_} at ({ax_}, {ay_})")

    # Create both windows as resizable
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MATH, cv2.WINDOW_NORMAL)

    # Size and position them side by side
    cv2.resizeWindow(WINDOW_MAIN, mw_, mh_)
    cv2.resizeWindow(WINDOW_MATH, aw_, ah_)
    cv2.moveWindow(WINDOW_MAIN, mx_, my_)
    cv2.moveWindow(WINDOW_MATH, ax_, ay_)

    # Two waitKey flushes: on Wayland/X11, moveWindow is asynchronous
    # and the window manager may need a hint to process the geometry.
    cv2.waitKey(1); cv2.waitKey(1)

    # Register mouse callback for pixel inspection
    cv2.setMouseCallback(WINDOW_MAIN, on_mouse)

    # Optional "always on top" using wmctrl if available (skip silently otherwise)
    if shutil.which("wmctrl"):
        for w in (WINDOW_MAIN, WINDOW_MATH):
            try:
                subprocess.Popen(["wmctrl", "-r", w, "-b", "add,above"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except Exception:
                pass

    STATE["start_time"] = time.time()
    print("=" * 64)
    print(" FocusGuard running.  1/2/3/4 panel | k kernel | m SE")
    print(" o dil/ero | l LAB/DEMO | s liveness | q quit")
    print("=" * 64)

    # try/finally guarantees cleanup
    try:
        while True:
            # --- 1. Read frame ---
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)      # mirror

            # --- 2. Downscale for CV work ---
            small = cv2.resize(frame, (WORK_W, WORK_H),
                               interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            STATE["frame_count"] += 1
            fc = STATE["frame_count"]

            # Map the mouse cursor position from canvas space to small space
            # (the math panels work in 480×360 coords)
            mxd, myd = STATE["mouse"]
            STATE["mouse_small_x"] = int(np.clip(
                mxd * WORK_W / max(mw_, 1), 0, WORK_W - 1))
            STATE["mouse_small_y"] = int(np.clip(
                myd * WORK_H / max(mh_, 1), 0, WORK_H - 1))

            # --- 3. Face detection every N frames ---
            if fc % DETECT_EVERY == 0:
                if kind == "dnn":
                    STATE["face_box"] = detect_face_dnn(detector, small)
                elif kind == "haar":
                    STATE["face_box"] = detect_face_haar(detector, gray_small)
                else:
                    STATE["face_box"] = None

            # --- 4. Mood pipeline ---
            mood = analyze_face(gray_small)

            # --- 5. Build FocusGuard canvas ---
            canvas = blank_canvas(mw_, mh_)
            # Leave 56px at top for the header strip
            fitted, xo, yo, fw_, fh_ = fit_frame(frame, mw_, mh_ - 56)
            canvas[56 + yo:56 + yo + fh_, xo:xo + fw_] = fitted
            fr = (xo, 56 + yo, fw_, fh_)
            # face_box is in small coords; draw_overlay scales it up once.
            draw_overlay(canvas, mood, fr, STATE["mouse"])

            # --- 6. Build Math Panel canvas ---
            math = get_math_panel(gray_small, fc, aw_, ah_)

            # --- 7. Show both windows ---
            cv2.imshow(WINDOW_MAIN, canvas)
            cv2.imshow(WINDOW_MATH, math)

            # Log focus for the session report
            if STATE["start_time"] is not None:
                t = time.time() - STATE["start_time"]
                STATE["focus_log"].append((t, mood["focus"]))

            # --- 8. Keyboard ---
            k = cv2.waitKey(1) & 0xFF
            if not handle_key(k):
                break

            # --- 9. Window-close detection ---
            # cv2.getWindowProperty returns < 1 when the user closes the
            # window via the X button.
            if cv2.getWindowProperty(WINDOW_MAIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.getWindowProperty(WINDOW_MATH, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        print("[focusguard] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[focusguard] Released.")


# ============================================================================
# SECTION 18 — SESSION REPORT
# ============================================================================
# When the user quits (via 'q' or window close), write a PNG showing
# the focus score over the whole session.
# ----------------------------------------------------------------------------

def session_report(path="session_report.png"):
    """Write a PNG report showing the focus score timeline."""
    log = STATE["focus_log"]
    if not log:
        print("[focusguard] no session data")
        return None

    # Unzip the log into two lists
    t = [a for a, _ in log]
    s = [b for _, b in log]

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
    for sp in ax.spines.values():
        sp.set_color("#5a5a6e")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(path, facecolor="#121218")
    plt.close(fig)
    print(f"[focusguard] report -> {path}")


# ============================================================================
# SECTION 19 — ENTRY POINT
# ============================================================================
# Standard Python idiom. try/finally ensures the session report is
# written even if main() raises an exception.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    finally:
        session_report()