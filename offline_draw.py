#!/usr/bin/env python3
"""Standalone cross-platform drawing recorder for BPP DrawAnything prompts.

This file deliberately does not import BPP, PyTorch, zarr, or pymunk.  It only
needs ``numpy`` and ``pygame`` so it can run on a Windows/macOS/Linux laptop
outside the BPP environment.  OpenCV is optional; when present, the 224x224
frames use the same ``cv2.resize`` interpolation as BPP.  The result is a
single compressed NPZ
containing the raw per-step RGB observations, raw actions, agent positions,
pen state, boundary angle, and the final drawing.  No model-specific encoding
or precomputed embedding is stored.

Typical use::

    pip install numpy pygame
    python offline_draw.py -o cat.npz --task-name cat

Controls::

    move the mouse close to the green cursor to grab it
    hold the left mouse button to draw; release it to stop drawing
    R       reset the current drawing
    D       finish and save
    Q / Esc quit without saving

The rendering and 10 Hz PD controller mirror
``behavior_prompting/train_network/env/draw/draw_env.py`` closely enough that
the NPZ can be converted to a BPP prompt without re-simulating the drawing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# Keep pygame's import banner out of the terminal.  Set this before importing.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if sys.platform == "win32":
    # Avoid Windows scaling the 512x512 board and mouse coordinates separately.
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - friendly message
    raise SystemExit("Missing dependency: numpy.  Install it with `pip install numpy`.") from exc

try:
    import pygame
except ImportError as exc:  # pragma: no cover - friendly message
    raise SystemExit("Missing dependency: pygame.  Install it with `pip install pygame`.") from exc

# OpenCV is optional.  BPP resizes 512x512 renders with cv2.resize; using it
# here makes the saved 224x224 prompt frames pixel-identical to BPP.  Without
# OpenCV the script falls back to pygame's cross-platform smoothscale.
try:
    import cv2 as _cv2
except ImportError:  # pragma: no cover - depends on the user's environment
    _cv2 = None

RESIZE_BACKEND = "cv2" if _cv2 is not None else "pygame-smoothscale"


FORMAT_NAME = "bpp-offline-draw-prompt"
FORMAT_VERSION = 1

WINDOW_SIZE = 512
RENDER_SIZE = 224
BOARD_LENGTH = 350
BOARD_HALF = BOARD_LENGTH / 2.0
BOARD_CENTER = WINDOW_SIZE / 2.0

CONTROL_HZ = 10
SIM_HZ = 100
K_P = 100.0
K_V = 20.0

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (48, 156, 54)
LIGHT_GRAY = (211, 211, 211)

AGENT_RADIUS = 15
AGENT_OUTLINE = 2
TRAIL_WIDTH = 12
TRAIL_DOT_RADIUS = 4
BOARD_LINE_WIDTH = 8
GRAB_DISTANCE = 30.0

# The PD cursor lags behind a fast mouse.  When the button is released (or D
# is pressed with the pen down), continue the last stroke toward the release
# position so the recorded drawing does not stop a few steps short.
SETTLE_MAX_STEPS = 10
SETTLE_DISTANCE = 1.0
SETTLE_SPEED = 1.0

REQUIRED_ARRAYS = (
    "image",
    "agent_pos",
    "pen_down",
    "action",
    "boundary_angle",
    "final_drawing",
)


def _point_int(point: Sequence[float]) -> Tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _surface_to_rgb(surface: "pygame.Surface") -> np.ndarray:
    """Return a C-contiguous HWC uint8 RGB array from a pygame surface."""
    # pygame.surfarray.array3d returns (width, height, 3).
    rgb = pygame.surfarray.array3d(surface)
    return np.ascontiguousarray(np.transpose(rgb, (1, 0, 2)), dtype=np.uint8)


def _rgb_to_surface(rgb: np.ndarray) -> "pygame.Surface":
    """Build a pygame surface from an HWC uint8 RGB array."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got {rgb.shape}")
    # pygame.surfarray.make_surface expects (width, height, 3).
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def _board_corners(angle: float) -> Tuple[Tuple[float, float], ...]:
    """Corners in the same order as DrawEnv: bottom-left, bottom-right, top-right, top-left."""
    corners = (
        (-BOARD_HALF, -BOARD_HALF),
        (BOARD_HALF, -BOARD_HALF),
        (BOARD_HALF, BOARD_HALF),
        (-BOARD_HALF, BOARD_HALF),
    )
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return tuple(
        (
            x * cos_a - y * sin_a + BOARD_CENTER,
            x * sin_a + y * cos_a + BOARD_CENTER,
        )
        for x, y in corners
    )


def _draw_fat_segment(
    surface: "pygame.Surface",
    color: Tuple[int, int, int],
    start: Sequence[float],
    end: Sequence[float],
    radius: float = 4.0,
) -> None:
    """Mirror pymunk's ``DrawOptions.draw_fat_segment``.

    Using the same polygon-plus-round-caps rasterization matters: a plain
    ``pygame.draw.line`` is one pixel narrower at rasterized edges, which
    visibly changes the board in the prompt observation.
    """
    p1 = _point_int(start)
    p2 = _point_int(end)
    thickness = round(max(1.0, radius * 2.0))
    pygame.draw.lines(surface, color, False, [p1, p2], thickness)
    if thickness <= 2:
        return

    orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
    if orthog[0] == 0 and orthog[1] == 0:
        return
    scale = radius / math.hypot(orthog[0], orthog[1])
    orthog[0] = round(orthog[0] * scale)
    orthog[1] = round(orthog[1] * scale)
    points = [
        (p1[0] - orthog[0], p1[1] - orthog[1]),
        (p1[0] + orthog[0], p1[1] + orthog[1]),
        (p2[0] + orthog[0], p2[1] + orthog[1]),
        (p2[0] - orthog[0], p2[1] - orthog[1]),
    ]
    pygame.draw.polygon(surface, color, points)
    pygame.draw.circle(surface, color, p1, round(radius))
    pygame.draw.circle(surface, color, p2, round(radius))


def _draw_board(surface: "pygame.Surface", boundary_angle: float) -> None:
    bl, br, tr, tl = _board_corners(boundary_angle)

    # DrawEnv adds the helper lines to the pymunk space before the walls.
    top_mid = ((tl[0] + tr[0]) / 2.0, (tl[1] + tr[1]) / 2.0)
    bottom_mid = ((bl[0] + br[0]) / 2.0, (bl[1] + br[1]) / 2.0)
    left_mid = ((tl[0] + bl[0]) / 2.0, (tl[1] + bl[1]) / 2.0)
    right_mid = ((tr[0] + br[0]) / 2.0, (tr[1] + br[1]) / 2.0)
    for start, end in ((tl, br), (tr, bl), (top_mid, bottom_mid), (left_mid, right_mid)):
        _draw_fat_segment(surface, LIGHT_GRAY, start, end)

    corners = (bl, br, tr, tl)
    for index in range(4):
        _draw_fat_segment(
            surface,
            RED if index == 0 else BLACK,
            corners[index],
            corners[(index + 1) % 4],
        )


def _draw_agent(surface: "pygame.Surface", position: Sequence[float]) -> None:
    point = _point_int(position)
    pygame.draw.circle(surface, GREEN, point, AGENT_RADIUS)
    pygame.draw.circle(surface, WHITE, point, AGENT_RADIUS, AGENT_OUTLINE)


class DrawSession:
    """Cursor dynamics, persistent trail, and observation rendering.

    The controller is the same explicit-Euler PD controller used by ``DrawEnv``:
    ten 100 Hz physics substeps per 10 Hz control step, ``k_p=100`` and
    ``k_v=20``.  The pymunk body in DrawEnv is kinematic, so no collision
    response is needed here.
    """

    def __init__(
        self,
        seed: int = 0,
        boundary_angle: float = 0.0,
        start_xy: Optional[Sequence[float]] = None,
        render_size: int = RENDER_SIZE,
    ) -> None:
        self.seed = int(seed)
        self.boundary_angle = float(boundary_angle)
        self.render_size = int(render_size)
        self.start_xy = None if start_xy is None else (float(start_xy[0]), float(start_xy[1]))
        self.reset()

    def reset(self) -> None:
        if self.start_xy is None:
            # Match DrawEnv.reset(): random positions in [50, 450) from a
            # RandomState seeded with the environment seed.
            random_state = np.random.RandomState(self.seed)
            self.start_xy = (
                float(random_state.randint(50, 450)),
                float(random_state.randint(50, 450)),
            )
        self.pos = np.asarray(self.start_xy, dtype=np.float64).copy()
        self.vel = np.zeros(2, dtype=np.float64)
        self.pen_down = False
        self.teleop = False
        self.trail = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        self.trail.fill(WHITE)
        self.trail.set_colorkey(WHITE)
        self.last_drawn: Optional[Tuple[int, int]] = None

    def try_engage(self, mouse_xy: Sequence[float], threshold: float = GRAB_DISTANCE) -> bool:
        if not self.teleop:
            distance = float(np.linalg.norm(np.asarray(mouse_xy, dtype=np.float64) - self.pos))
            if distance < threshold:
                self.teleop = True
        return self.teleop

    def step(self, action: Sequence[float]) -> None:
        """Apply one 10 Hz action: ``[target_x, target_y, pen_down]``."""
        target = np.asarray(action[:2], dtype=np.float64)
        pen_down = bool(float(action[2]) > 0.5)
        self.pen_down = pen_down

        dt = 1.0 / SIM_HZ
        for _ in range(SIM_HZ // CONTROL_HZ):
            acceleration = K_P * (target - self.pos) + K_V * (0.0 - self.vel)
            self.vel += acceleration * dt
            self.pos += self.vel * dt

        if pen_down:
            point = _point_int(self.pos)
            if self.last_drawn is not None:
                pygame.draw.line(self.trail, BLUE, self.last_drawn, point, TRAIL_WIDTH)
            pygame.draw.circle(self.trail, BLUE, point, TRAIL_DOT_RADIUS)
            self.last_drawn = point
        else:
            self.last_drawn = None

    def render_canvas(self) -> "pygame.Surface":
        """Render the 512x512 observation exactly as DrawEnv does."""
        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill(WHITE)
        _draw_board(canvas, self.boundary_angle)
        canvas.blit(self.trail, (0, 0))
        _draw_agent(canvas, self.pos)
        return canvas

    def observation_from_canvas(self, canvas: "pygame.Surface") -> np.ndarray:
        if self.render_size == WINDOW_SIZE:
            return _surface_to_rgb(canvas)
        if _cv2 is not None:
            # Same interpolation as DrawEnv._render_frame; cv2.resize is
            # deterministic and gives pixel-identical 224x224 frames.
            return _cv2.resize(
                _surface_to_rgb(canvas),
                (self.render_size, self.render_size),
                interpolation=_cv2.INTER_LINEAR,
            )
        scaled = pygame.transform.smoothscale(canvas, (self.render_size, self.render_size))
        return _surface_to_rgb(scaled)

    def observation(self) -> np.ndarray:
        return self.observation_from_canvas(self.render_canvas())

    def final_drawing(self) -> np.ndarray:
        """The 512x512 white/blue drawing canvas without board or cursor."""
        canvas = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
        canvas.fill(WHITE)
        canvas.blit(self.trail, (0, 0))
        return _surface_to_rgb(canvas)


class Recorder:
    """Accumulates raw per-step data for one demonstration."""

    def __init__(self) -> None:
        self.images = []
        self.agent_pos = []
        self.pen_down = []
        self.actions = []
        self.boundary_angles = []

    @property
    def n_steps(self) -> int:
        return len(self.actions)

    def clear(self) -> None:
        self.images.clear()
        self.agent_pos.clear()
        self.pen_down.clear()
        self.actions.clear()
        self.boundary_angles.clear()

    def add(self, session: DrawSession, action: Sequence[float], canvas: "pygame.Surface") -> None:
        # State at time t (before applying action t), matching demo_draw.py.
        self.images.append(session.observation_from_canvas(canvas))
        self.agent_pos.append(session.pos.astype(np.float32).copy())
        self.pen_down.append(np.array([1.0 if session.pen_down else 0.0], dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.boundary_angles.append(np.float32(session.boundary_angle))

    def arrays(self, session: DrawSession) -> Dict[str, np.ndarray]:
        if self.n_steps == 0:
            raise ValueError("No steps were recorded")
        return {
            "image": np.stack(self.images, axis=0).astype(np.uint8, copy=False),
            "agent_pos": np.stack(self.agent_pos, axis=0).astype(np.float32, copy=False),
            "pen_down": np.stack(self.pen_down, axis=0).astype(np.float32, copy=False),
            "action": np.stack(self.actions, axis=0).astype(np.float32, copy=False),
            "boundary_angle": np.asarray(self.boundary_angles, dtype=np.float32),
            "final_drawing": session.final_drawing().astype(np.uint8, copy=False),
        }


def _settle_pen_down(
    session: DrawSession,
    recorder: Recorder,
    target_xy: Sequence[float],
) -> None:
    """Move the pen to the release/finish position with the pen still down.

    DrawEnv's cursor is a PD controller, so during a fast stroke the cursor can
    lag the mouse by tens of pixels.  Without this settling phase, releasing the
    button makes the drawing end where the lagging cursor was, not where the
    user released it.  Settling records normal 10 Hz actions, so the saved
    prompt stays consistent with the rendered frames.
    """
    target = np.asarray(target_xy, dtype=np.float64)
    for _ in range(SETTLE_MAX_STEPS):
        distance = float(np.linalg.norm(target - session.pos))
        speed = float(np.linalg.norm(session.vel))
        if distance <= SETTLE_DISTANCE and speed <= SETTLE_SPEED:
            break
        action = np.array([target[0], target[1], 1.0], dtype=np.float32)
        canvas = session.render_canvas()
        recorder.add(session, action, canvas)
        session.step(action)


def normalize_task_name(raw: str) -> str:
    """Return the BPP task name (``draw ...``) while validating its case."""
    text = raw.strip()
    if not text:
        raise ValueError("task name cannot be empty")
    lower = text.lower()
    if lower.startswith("draw "):
        suffix = text[5:].strip()
    elif lower.startswith("draw_"):
        suffix = text[5:].strip()
    else:
        suffix = text
    if not suffix:
        raise ValueError("task name cannot be just 'draw'")
    if suffix.upper() != suffix and suffix.lower() != suffix:
        raise ValueError(
            "BPP task names after 'draw ' must be all uppercase or all lowercase; "
            f"got {suffix!r}"
        )
    return f"draw {suffix}"


def _default_task_name(output: Path) -> str:
    stem = output.stem
    if stem.lower().startswith("draw_"):
        stem = stem[5:]
    return normalize_task_name(stem or "prompt")


def _build_meta(task_name: str, session: DrawSession, n_steps: int) -> Dict[str, object]:
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "task_name": task_name,
        "control_hz": CONTROL_HZ,
        "canvas_size": WINDOW_SIZE,
        "render_size": session.render_size,
        "board_length": BOARD_LENGTH,
        "boundary_angle": session.boundary_angle,
        "resize_backend": RESIZE_BACKEND,
        "seed": session.seed,
        "steps": int(n_steps),
        "duration_seconds": float(n_steps) / CONTROL_HZ,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "generator": "experiments/offline_prompt/offline_draw.py",
    }


def save_prompt(
    output: Path,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
    force: bool = False,
) -> None:
    output = output.expanduser()
    if output.suffix.lower() != ".npz":
        output = output.with_name(output.name + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists; pass --force to overwrite it before drawing"
        )

    payload = dict(arrays)
    payload["task_name"] = np.asarray(str(meta["task_name"]))
    payload["format"] = np.asarray(FORMAT_NAME)
    payload["version"] = np.asarray(FORMAT_VERSION, dtype=np.int64)
    payload["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))

    # Write to a sibling temporary file and atomically replace the target, so
    # an interrupted save cannot leave a half-written prompt.
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".npz.tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temp_name, output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def load_prompt(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load and validate an offline-draw prompt without allowing pickle."""
    path = path.expanduser()
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    missing = [key for key in REQUIRED_ARRAYS if key not in data]
    if missing:
        raise ValueError(f"Missing arrays in {path}: {missing}")

    meta: Dict[str, object] = {}
    if "meta_json" in data:
        meta = json.loads(str(data["meta_json"].item()))

    image = data["image"]
    actions = data["action"]
    if image.ndim != 4 or image.shape[1:] != (RENDER_SIZE, RENDER_SIZE, 3):
        raise ValueError(
            f"image must be (T, {RENDER_SIZE}, {RENDER_SIZE}, 3), got {image.shape}"
        )
    n_steps = image.shape[0]
    if n_steps == 0:
        raise ValueError("prompt contains no steps")
    expected = {
        "agent_pos": (n_steps, 2),
        "pen_down": (n_steps, 1),
        "action": (n_steps, 3),
        "boundary_angle": (n_steps,),
    }
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {data[key].shape}")
    if data["final_drawing"].ndim != 3 or data["final_drawing"].shape != (WINDOW_SIZE, WINDOW_SIZE, 3):
        raise ValueError(
            f"final_drawing must be (512, 512, 3), got {data['final_drawing'].shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if data["final_drawing"].dtype != np.uint8:
        raise ValueError(f"final_drawing must be uint8, got {data['final_drawing'].dtype}")
    if not np.isfinite(actions).all():
        raise ValueError("action contains non-finite values")
    if not np.isfinite(data["agent_pos"]).all():
        raise ValueError("agent_pos contains non-finite values")
    if not np.isfinite(data["boundary_angle"]).all():
        raise ValueError("boundary_angle contains non-finite values")
    if actions[:, :2].min() < 0 or actions[:, :2].max() > WINDOW_SIZE:
        raise ValueError("action x/y must be in [0, 512]")
    if not np.isin(data["pen_down"], (0.0, 1.0)).all():
        raise ValueError("pen_down must only contain 0.0 and 1.0")
    if not np.isin(actions[:, 2], (0.0, 1.0)).all():
        raise ValueError("action pen channel must only contain 0.0 and 1.0")

    return data, meta


def _print_summary(data: Dict[str, np.ndarray], meta: Dict[str, object]) -> None:
    image = data["image"]
    actions = data["action"]
    final = data["final_drawing"]
    drawing_pixels = int(
        np.count_nonzero((final[:, :, 2] > 200) & (final[:, :, 0] < 50) & (final[:, :, 1] < 50))
    )
    print(f"format:       {meta.get('format', data.get('format', '?'))}")
    print(f"version:      {meta.get('version', data.get('version', '?'))}")
    print(f"task name:    {meta.get('task_name', data.get('task_name', '?'))}")
    print(f"steps:        {image.shape[0]} ({image.shape[0] / CONTROL_HZ:.1f} s at {CONTROL_HZ} Hz)")
    print(f"image:        {image.shape} {image.dtype}")
    print(f"agent_pos:    {data['agent_pos'].shape} {data['agent_pos'].dtype}")
    print(f"pen_down:     {data['pen_down'].shape} {data['pen_down'].dtype}")
    print(f"action:       {actions.shape} {actions.dtype}")
    print(f"final:        {final.shape} {final.dtype}, blue pixels={drawing_pixels}")
    if meta:
        print(f"created:      {meta.get('created_utc', '?')}")
        print(f"platform:     {meta.get('platform', '?')}")


def _save_preview(path: Path, rgb: np.ndarray) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(_rgb_to_surface(rgb), str(path))
    print(f"preview saved: {path}")


def _mouse_in_canvas() -> Tuple[int, int]:
    x, y = pygame.mouse.get_pos()
    return max(0, min(WINDOW_SIZE - 1, int(x))), max(0, min(WINDOW_SIZE - 1, int(y)))


def run_interactive(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser()
    if output.suffix.lower() != ".npz":
        output = output.with_name(output.name + ".npz")
    if output.exists() and not args.force:
        print(
            f"Refusing to overwrite {output}; pass --force or choose another -o path.",
            file=sys.stderr,
        )
        return 2

    try:
        task_name = normalize_task_name(args.task_name) if args.task_name else _default_task_name(output)
    except ValueError as exc:
        print(f"Invalid task name: {exc}", file=sys.stderr)
        print("Use --task-name with an all-uppercase or all-lowercase name.", file=sys.stderr)
        return 2
    start_xy = None
    if args.start is not None:
        start_xy = (float(args.start[0]), float(args.start[1]))
        if not (0 <= start_xy[0] < WINDOW_SIZE and 0 <= start_xy[1] < WINDOW_SIZE):
            print("--start coordinates must be inside the 512x512 canvas", file=sys.stderr)
            return 2

    boundary_angle = math.radians(float(args.boundary_angle_deg))
    session = DrawSession(seed=args.seed, boundary_angle=boundary_angle, start_xy=start_xy)
    recorder = Recorder()

    pygame.display.init()
    try:
        screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    except pygame.error as exc:
        print(f"Could not open a window: {exc}", file=sys.stderr)
        print("On a headless machine, use --selftest or --inspect instead.", file=sys.stderr)
        pygame.display.quit()
        return 1

    pygame.display.set_caption(
        "BPP offline draw | grab cursor | hold LMB to draw | R retry | D save | Q quit"
    )
    clock = pygame.time.Clock()
    print(f'Offline draw for task "{task_name}"')
    print(f"Output: {output}")
    if args.force and output.exists():
        print("--force: the existing prompt will be overwritten after you press D.")
    print("Move the mouse near the green cursor to grab it.")
    print("Hold the left mouse button to draw; release it to stop drawing.")
    print("R retry | D done and save | Q/Esc quit without saving")

    done = False
    running = True
    announced_grab = False
    finish_requested = False
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        session.reset()
                        recorder.clear()
                        announced_grab = False
                        finish_requested = False
                        print("Drawing reset.")
                    elif event.key == pygame.K_d:
                        if recorder.n_steps > 0:
                            finish_requested = True
                        else:
                            print("Nothing to save yet; grab the cursor and draw first.")

            if not running:
                break

            mouse = _mouse_in_canvas()
            session.try_engage(mouse)
            if session.teleop and not announced_grab:
                announced_grab = True
                print(f"Cursor grabbed; recording at {CONTROL_HZ} Hz.")

            canvas = session.render_canvas()
            if session.teleop:
                pressed = pygame.mouse.get_pressed()
                pen_down = 1.0 if (pressed and pressed[0]) else 0.0
                released = session.pen_down and pen_down == 0.0

                # On release, finish the stroke at the mouse position before
                # recording the pen-up action.
                if released:
                    _settle_pen_down(session, recorder, mouse)
                    canvas = session.render_canvas()

                # If D is pressed while the pen is already up, keep the last
                # pen-up action and finish without adding another action.
                if not (finish_requested and not pen_down and not released):
                    action = np.array([mouse[0], mouse[1], pen_down], dtype=np.float32)
                    recorder.add(session, action, canvas)
                    session.step(action)

                    if finish_requested and pen_down:
                        # Finish the last stroke at the mouse position, then
                        # record a clean pen-up action at the same place.
                        _settle_pen_down(session, recorder, mouse)
                        canvas = session.render_canvas()
                        action = np.array([mouse[0], mouse[1], 0.0], dtype=np.float32)
                        recorder.add(session, action, canvas)
                        session.step(action)

                canvas = session.render_canvas()

            screen.blit(canvas, (0, 0))
            pygame.display.flip()

            if finish_requested:
                done = True
                break

            clock.tick(CONTROL_HZ)
    finally:
        pygame.display.quit()

    if not done:
        print(f"Quit without saving. Discarded {recorder.n_steps} recorded steps.")
        return 0
    if recorder.n_steps == 0:
        print("Nothing was recorded; no file written.", file=sys.stderr)
        return 1
    if recorder.n_steps < CONTROL_HZ:
        print(
            f"Warning: only {recorder.n_steps} steps recorded; a prompt chunk is "
            f"{CONTROL_HZ} steps and very short prompts may be weak.",
            file=sys.stderr,
        )

    arrays = recorder.arrays(session)
    meta = _build_meta(task_name, session, recorder.n_steps)
    try:
        save_prompt(output, arrays, meta, force=args.force)
    except (OSError, ValueError) as exc:
        print(f"Could not save prompt: {exc}", file=sys.stderr)
        return 1

    print(
        f'Saved {recorder.n_steps} steps ({recorder.n_steps / CONTROL_HZ:.1f} s) '
        f'for "{task_name}" to {output}'
    )
    if args.preview:
        _save_preview(Path(args.preview), arrays["final_drawing"])
    return 0


def run_selftest() -> int:
    """Headless smoke test used by developers and CI."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    session = DrawSession(seed=7, boundary_angle=0.0, start_xy=(256, 256))
    recorder = Recorder()
    # A small square, one control step per side point.
    for step in range(60):
        if step < 10:
            target = (256 + 4 * step, 256)
            pen = 0.0
        elif step < 25:
            target = (400, 256 + 8 * (step - 10))
            pen = 1.0
        elif step < 40:
            target = (400 - 8 * (step - 25), 376)
            pen = 1.0
        elif step < 55:
            target = (280, 376 - 8 * (step - 40))
            pen = 1.0
        else:
            target = (280, 256)
            pen = 0.0
        action = np.array([target[0], target[1], pen], dtype=np.float32)
        canvas = session.render_canvas()
        recorder.add(session, action, canvas)
        session.step(action)

    arrays = recorder.arrays(session)
    meta = _build_meta("draw selftest", session, recorder.n_steps)
    if arrays["image"].shape != (60, RENDER_SIZE, RENDER_SIZE, 3):
        raise AssertionError(arrays["image"].shape)
    if arrays["action"].shape != (60, 3):
        raise AssertionError(arrays["action"].shape)
    if not np.isfinite(arrays["action"]).all():
        raise AssertionError("non-finite action")
    blue = (
        (arrays["final_drawing"][:, :, 2] > 200)
        & (arrays["final_drawing"][:, :, 0] < 50)
        & (arrays["final_drawing"][:, :, 1] < 50)
    )
    if not blue.any():
        raise AssertionError("final drawing contains no blue strokes")
    if not np.isfinite(arrays["agent_pos"]).all():
        raise AssertionError("non-finite agent position")

    with tempfile.TemporaryDirectory(prefix="bpp-offline-draw-selftest-") as temp_dir:
        path = Path(temp_dir) / "selftest.npz"
        save_prompt(path, arrays, meta)
        loaded, loaded_meta = load_prompt(path)
        for key in REQUIRED_ARRAYS:
            if not np.array_equal(loaded[key], arrays[key]):
                raise AssertionError(f"round-trip mismatch for {key}")
        if loaded_meta.get("task_name") != "draw selftest":
            raise AssertionError(loaded_meta)

    print(
        "selftest ok: "
        f"{recorder.n_steps} steps, image {arrays['image'].shape}, "
        f"final drawing {arrays['final_drawing'].shape}"
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw an offline DrawAnything prompt and save raw data to one NPZ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="drawing_prompt.npz",
        help="output .npz path (a .npz suffix is added when missing)",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="BPP task name, e.g. cat or 'draw CAT'; defaults to the output stem",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the initial cursor position")
    parser.add_argument(
        "--start",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="explicit initial cursor position in 512x512 coordinates",
    )
    parser.add_argument(
        "--boundary-angle-deg",
        type=float,
        default=0.0,
        help="board rotation in degrees; keep 0 for a normal BPP prompt",
    )
    parser.add_argument("--force", action="store_true", help="overwrite the output file")
    parser.add_argument("--preview", default=None, help="also write the final drawing as a PNG")
    parser.add_argument("--inspect", default=None, help="validate and print an existing .npz, then exit")
    parser.add_argument("--selftest", action="store_true", help="run a headless smoke test and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()
    if args.inspect:
        try:
            data, meta = load_prompt(Path(args.inspect))
        except (OSError, ValueError, KeyError) as exc:
            print(f"Invalid prompt: {exc}", file=sys.stderr)
            return 1
        _print_summary(data, meta)
        if args.preview:
            _save_preview(Path(args.preview), data["final_drawing"])
        return 0
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
