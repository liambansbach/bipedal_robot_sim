from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import genesis as gs

from robot_gym.utils.helpers import class_to_dict


GENESIS_SUBTERRAINS = [
    "flat_terrain",
    "fractal_terrain",
    "random_uniform_terrain",
    "sloped_terrain",
    "pyramid_sloped_terrain",
    "discrete_obstacles_terrain",
    "wave_terrain",
    "stairs_terrain",
    "pyramid_stairs_terrain",
    "stepping_stones_terrain",
]


@dataclass
class TerrainSpec:
    terrain_type: str
    morph: Any
    surface: Any
    subterrain_types: Optional[List[List[str]]] = None
    metadata: Optional[Dict[str, Any]] = None


def _normalize_probs(options: List[str], probs: List[float]) -> np.ndarray:
    """Normalizes a list of probabilities and checks for validity.
    """
    if len(options) == 0:
        raise ValueError("cfg.terrain.options must not be empty when mode='random'.")

    if len(probs) == 0:
        return np.ones(len(options), dtype=np.float64) / float(len(options))

    if len(options) != len(probs):
        raise ValueError(
            f"terrain.options and terrain.probs must have same length, got "
            f"{len(options)} and {len(probs)}."
        )

    p = np.asarray(probs, dtype=np.float64)
    if np.any(p < 0.0):
        raise ValueError("terrain.probs must be non-negative.")
    s = p.sum()
    if s <= 0.0:
        raise ValueError("terrain.probs must sum to a positive value.")
    return p / s


def _resolve_top_level_terrain_type(cfg) -> str:
    """Resolves the top-level terrain type based on cfg.terrain.mode.
    If mode is 'random', it randomly selects one terrain type from cfg.terrain.options using probabilities from cfg.terrain.probs.
    If mode is not 'random', it returns mode directly (after checking for validity).
    """
    mode = str(cfg.mode)

    if mode == "random":
        options = list(cfg.options)
        probs = _normalize_probs(options, list(cfg.probs))
        selected = str(np.random.choice(options, p=probs))
        return selected

    return mode


def _get_global_terrain_params(cfg) -> Dict[str, Dict]:
    """
    Reads cfg.terrain.terrain_kwargs.<terrain_name> nested classes
    and converts them into the dict format expected by Genesis.
    """
    terrain_kwargs = {}

    if not hasattr(cfg, "terrain_kwargs"):
        return terrain_kwargs

    for terrain_name in GENESIS_SUBTERRAINS:
        if hasattr(cfg.terrain_kwargs, terrain_name):
            terrain_kwargs[terrain_name] = class_to_dict(
                getattr(cfg.terrain_kwargs, terrain_name)
            )

    return terrain_kwargs


def _build_single_type_grid(
    terrain_type: str,
    n_subterrains: Tuple[int, int],
) -> List[List[str]]:
    """
    Builds a grid of subterrains of a single type.
    """
    n_x, n_y = n_subterrains
    return [[terrain_type for _ in range(n_y)] for _ in range(n_x)]


def _build_mixed_grid(cfg) -> List[List[str]]:
    """
    Tile-wise random terrain selection.
    Useful when cfg.mode == "mixed".
    """
    n_x, n_y = cfg.n_subterrains
    options = list(cfg.mixed.options)
    probs = _normalize_probs(options, list(cfg.mixed.probs))

    grid: List[List[str]] = []
    for _ in range(n_x):
        row = [str(np.random.choice(options, p=probs)) for _ in range(n_y)]
        grid.append(row)
    return grid


def _apply_spawn_flat_zone(
    grid: List[List[str]],
    spawn_flat_radius_sub: int,
) -> None:
    """
    Replaces center tiles by flat terrain.
    radius 0 -> only center tile
    radius 1 -> 3x3 center area
    """
    if spawn_flat_radius_sub < 0:
        return

    n_x = len(grid)
    n_y = len(grid[0]) if n_x > 0 else 0
    c_x, c_y = n_x // 2, n_y // 2

    for i in range(n_x):
        for j in range(n_y):
            di = abs(i - c_x)
            dj = abs(j - c_y)
            if max(di, dj) <= spawn_flat_radius_sub:
                grid[i][j] = "flat_terrain"


def _apply_border_flat(grid: List[List[str]]) -> None:
    """
    Replaces border tiles by flat terrain.
    """
    n_x = len(grid)
    n_y = len(grid[0]) if n_x > 0 else 0
    if n_x == 0 or n_y == 0:
        return

    for j in range(n_y):
        grid[0][j] = "flat_terrain"
        grid[n_x - 1][j] = "flat_terrain"

    for i in range(n_x):
        grid[i][0] = "flat_terrain"
        grid[i][n_y - 1] = "flat_terrain"


def _compute_centered_terrain_pos(
    n_subterrains: Tuple[int, int],
    subterrain_size: Tuple[float, float],
) -> Tuple[float, float, float]:
    """
    center the terrain, so the flat patch is actually centered around world origin
    """
    n_x, n_y = n_subterrains
    sx, sy = subterrain_size
    total_x = n_x * sx
    total_y = n_y * sy
    return (-0.5 * total_x, -0.5 * total_y, 0.0)


def _build_subterrain_grid(cfg, selected_type: str) -> List[List[str]]:
    if selected_type == "mixed":
        grid = _build_mixed_grid(cfg)
    elif selected_type in GENESIS_SUBTERRAINS:
        grid = _build_single_type_grid(selected_type, cfg.n_subterrains)
    else:
        raise ValueError(
            f"Unsupported terrain type '{selected_type}'. "
            f"Supported: 'plane', 'mixed', or one of {GENESIS_SUBTERRAINS}"
        )

    if getattr(cfg, "spawn_flat_radius_sub", -1) >= 0:
        _apply_spawn_flat_zone(grid, int(cfg.spawn_flat_radius_sub))

    if getattr(cfg, "border_flat", False):
        _apply_border_flat(grid)

    return grid


def build_terrain_spec(cfg) -> TerrainSpec:
    """
    Main entry point used by LeggedRobot.
    """
    selected_type = _resolve_top_level_terrain_type(cfg)

    ground_surface = gs.surfaces.Default(
        color=tuple(getattr(cfg, "color", (0.5, 0.5, 0.5)))
    )

    # ------------------------------------------------------------------
    # 1) Simple infinite plane
    # ------------------------------------------------------------------
    if selected_type == "plane":
        return TerrainSpec(
            terrain_type="plane",
            morph=gs.morphs.Plane(),
            surface=ground_surface,
            metadata={"mode": "plane"},
        )

    # ------------------------------------------------------------------
    # 2) User-provided height field
    # ------------------------------------------------------------------
    if selected_type == "heightfield":
        if not hasattr(cfg, "heightfield") or cfg.heightfield is None:
            raise ValueError("cfg.terrain.mode='heightfield' but cfg.terrain.heightfield is None.")

        pos = getattr(cfg, "pos", None)
        if pos is None:
            pos = (0.0, 0.0, 0.0)

        morph = gs.morphs.Terrain(
            height_field=cfg.heightfield,
            horizontal_scale=float(cfg.horizontal_scale),
            vertical_scale=float(cfg.vertical_scale),
            pos=tuple(pos),
            name=getattr(cfg, "name", None),
        )

        return TerrainSpec(
            terrain_type="heightfield",
            morph=morph,
            surface=ground_surface,
            metadata={"mode": "heightfield"},
        )

    # ------------------------------------------------------------------
    # 3) Procedural Genesis terrain
    # ------------------------------------------------------------------
    if selected_type not in GENESIS_SUBTERRAINS and selected_type != "mixed":
        raise ValueError(
            f"Unknown terrain mode/type '{selected_type}'. "
            f"Use 'plane', 'heightfield', 'mixed', 'random', or a Genesis terrain type."
        )

    subterrain_types = _build_subterrain_grid(cfg, selected_type)

    pos = getattr(cfg, "pos", None)
    if pos is None:
        pos = _compute_centered_terrain_pos(cfg.n_subterrains, cfg.subterrain_size)

    morph = gs.morphs.Terrain(
        pos=tuple(pos),
        n_subterrains=tuple(cfg.n_subterrains),
        subterrain_size=tuple(cfg.subterrain_size),
        horizontal_scale=float(cfg.horizontal_scale),
        vertical_scale=float(cfg.vertical_scale),
        subterrain_types=subterrain_types,
        randomize=bool(getattr(cfg, "randomize", False)),
        name=getattr(cfg, "name", None),
        subterrain_parameters=_get_global_terrain_params(cfg),
    )

    return TerrainSpec(
        terrain_type=selected_type,
        morph=morph,
        surface=ground_surface,
        subterrain_types=subterrain_types,
        metadata={
            "mode": selected_type,
            "n_subterrains": tuple(cfg.n_subterrains),
            "subterrain_size": tuple(cfg.subterrain_size),
            "horizontal_scale": float(cfg.horizontal_scale),
            "vertical_scale": float(cfg.vertical_scale),
            "pos": tuple(pos),
        },
    )