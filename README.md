# FocusGuard — Live Classical Computer Vision Demo

A real-time webcam demo that makes the mathematics of classical computer
vision *visible at the pixel level*. Two windows: a mood/focus detector
(the product), and a math panel that shows the exact arithmetic the
detector is running (the glass box). Built for teaching morphology,
edge detection, and segmentation from Szeliski Ch. 3–4.

## What it demonstrates

- **Binary morphology** — dilation as `sup`, erosion as `inf`, opening
  and closing as idempotent composites
- **Structuring element design** — square, cross, ellipse, line, diagonal
- **Grayscale morphology** — local max / local min over a footprint
- **First-order edge detection** — Sobel, Prewitt, Roberts, Laplacian
- **Canny pipeline** — Gaussian → gradient → NMS → hysteresis
- **Otsu thresholding** — `T* = argmax σ_B²(T)` with the curve drawn live
- **Hough transform** — `ρ = x cos θ + y sin θ`, with the accumulator
  visible as it fills
- **Convolution** — elementwise product + sum, shown on real pixels
- **Application** — mood / focus / liveness detection from the same math

## Requirements

- Debian Linux (or any Linux with V4L2)
- Python 3.10+
- A webcam at `/dev/video0`
- VS Code with the `ms-toolsai.jupyter` extension (for notebook mode)

## Setup

```bash
cd focusguard
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or, if you just want to run it, use the launcher:

```bash
chmod +x run_demo.sh
./run_demo.sh
```

The launcher creates the venv on first run, installs dependencies, and
starts the demo.

## How to run

### Option A — from the terminal (recommended)

```bash
source .venv/bin/activate
python focusguard.py
```

### Option B — from VS Code Jupyter

1. Open `FocusGuard.ipynb` in VS Code
2. `Ctrl+Shift+P` → **Jupyter: Select Kernel** → pick `.venv`
3. Run the first cell (it does `%run focusguard.py`)

The notebook is a thin wrapper. All the logic lives in `focusguard.py`.

## Controls

| Key | Action |
|---|---|
| `1` | Math panel → Kernel Convolver |
| `2` | Math panel → Morphology Stamper |
| `3` | Math panel → Otsu Curve |
| `4` | Math panel → Hough Accumulator |
| `k` | Cycle kernel (Sobel X/Y, Prewitt X/Y, Roberts, Laplacian, Gaussian) |
| `m` | Cycle structuring element (square, cross, ellipse, h-line, v-line, diagonal) |
| `o` | Toggle dilate ↔ erode |
| `l` | Toggle LAB mode ↔ DEMO mode |
| `s` | Toggle liveness detection |
| `q` | Quit (also runs the session report) |

## Slide-to-panel mapping

| Slide | Concept | Panel |
|---|---|---|
| 3–4 | Set theory, binary dilation | 2 |
| 5 | Binary erosion | 2 |
| 6 | Opening & closing | 2 |
| 7 | Grayscale morphology | 2 |
| 8 | Hit-or-miss | 2 (composite SE) |
| 9 | Skeletonization | 2 (diagonal SE) |
| 10 | SE design | 2 (cycle `m`) |
| 11–13 | Gradient, Sobel | 1 (`k` → sobel_x) |
| 14 | Prewitt, Roberts | 1 (`k` → prewitt_x, roberts_x) |
| 15–18 | Kernel comparison | 1 (cycle `k`) |
| 19 | Canny pipeline | 4 + FocusGuard overlay |
| 20 | Harris corners | FocusGuard overlay (mouth corners) |
| 22–23 | Segmentation definition | FocusGuard ROI split |
| 24 | Adaptive threshold | FocusGuard eye ROI |
| 25–26 | Otsu | 3 |
| 27 | Region growing | FocusGuard pupil seed |
| 28 | Watershed | FocusGuard eye+brow split |
| 29 | Hough | 4 |
| 30 | Pipeline | Full FocusGuard window |

## Troubleshooting (VS Code)

**Kernel not found.** `Ctrl+Shift+P` → **Jupyter: Select Kernel** → pick
the `.venv` interpreter inside the `focusguard/` folder. VS Code often
defaults to the system Python, which won't have `cv2`.

**cv2 window is blank or black.** Run `python focusguard.py` from the
VS Code integrated terminal instead of the notebook. Some VS Code
notebook renderers steal the Qt/GTK event loop that `cv2.imshow` needs.

**Cell won't stop.** Click the square Stop icon in the cell's toolbar.
The main loop has a `try/finally` that releases the camera on any exit
path, including interrupts. If that fails, `Ctrl+Shift+P` →
**Developer: Reload Window**.

**Camera busy on next run.** A previous run didn't clean up. Either
wait 10 seconds, or run `pkill -f focusguard` from the terminal.

**DNN model files missing.** The face detector falls back to Haar
cascades automatically — the demo still runs, just with a slightly
weaker detector. No action needed.

**Low FPS.** Increase the frame-skip for face detection. In
`focusguard.py`, find `DETECT_EVERY = 3` and change it to `5`.

## Files

```
focusguard/
├── README.md            ← this file
├── requirements.txt     ← pip dependencies (3 packages)
├── focusguard.py        ← the demo (source of truth)
├── FocusGuard.ipynb     ← thin Jupyter wrapper (%run focusguard.py)
├── run_demo.sh          ← one-command launcher
└── assets/
    └── .gitkeep         ← for your screenshots/recordings
```

## Closing note

> Computer vision is not a black box. It is
> `-50 + 0 + 55 - 140 + 0 + 180 - 65 + 0 + 80 = 60`, computed 30 times
> per second, on your face.