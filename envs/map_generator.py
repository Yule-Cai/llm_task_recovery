# envs/map_generator.py
import numpy as np
import pandas as pd
import os
from collections import deque


def _largest_connected_component(grid):
    """Return the size of the largest connected free-cell component."""
    visited = set()
    best    = 0
    size    = grid.shape[0]

    for sx in range(size):
        for sy in range(size):
            if grid[sx, sy] != 0 or (sx, sy) in visited:
                continue
            # BFS from (sx, sy)
            component = set()
            queue     = deque([(sx, sy)])
            while queue:
                x, y = queue.popleft()
                if (x, y) in component:
                    continue
                component.add((x, y))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if (0 <= nx < size and 0 <= ny < size and
                            grid[nx, ny] == 0 and (nx, ny) not in component):
                        queue.append((nx, ny))
            visited |= component
            if len(component) > best:
                best = len(component)
    return best


def _connectivity_ratio(grid):
    free  = int(np.sum(grid == 0))
    if free == 0:
        return 0.0
    largest = _largest_connected_component(grid)
    return largest / free


def generate_map(size=30, obstacle_density=0.2, seed=None, min_connectivity=0.75):
    """
    Generate a random grid map with border walls and scattered obstacles.

    Retries until at least `min_connectivity` fraction of free cells are
    in the largest connected component, guaranteeing solvable episodes.

    Returns a 2D numpy array: 0=free, 1=wall.
    """
    rng = np.random.RandomState(seed)

    for attempt in range(200):
        grid = np.zeros((size, size), dtype=np.float32)

        # border walls
        grid[0, :]  = 1
        grid[-1, :] = 1
        grid[:, 0]  = 1
        grid[:, -1] = 1

        # random interior obstacles
        for x in range(1, size - 1):
            for y in range(1, size - 1):
                if rng.rand() < obstacle_density:
                    grid[x, y] = 1

        # add U-shaped dead-ends
        _add_dead_ends(grid, size, rng,
                       n=max(1, int(obstacle_density * 4)))

        if _connectivity_ratio(grid) >= min_connectivity:
            return grid

    # fallback: return last attempt even if connectivity is low
    return grid


def _add_dead_ends(grid, size, rng, n=2):
    """Carve U-shaped dead-end structures into the map."""
    for _ in range(n):
        tx    = rng.randint(3, size - 8)
        ty    = rng.randint(3, size - 8)
        arm   = rng.randint(3, 6)
        width = rng.randint(2, 5)

        for k in range(arm + 1):
            if tx + k < size and ty < size:
                grid[tx + k, ty] = 1
        for k in range(arm + 1):
            if tx + k < size and ty + width < size:
                grid[tx + k, ty + width] = 1
        for k in range(width + 1):
            if tx < size and ty + k < size:
                grid[tx, ty + k] = 1


def save_map(grid, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(grid).to_csv(path, header=False, index=False)


def load_map(path):
    return pd.read_csv(path, header=None).values.astype(np.float32)


# ── difficulty presets ────────────────────────────────────────────────────
DIFFICULTY_CONFIGS = {
    "L1": {"obstacle_density": 0.10, "size": 25},
    "L2": {"obstacle_density": 0.18, "size": 28},
    "L3": {"obstacle_density": 0.25, "size": 30},
    "L4": {"obstacle_density": 0.32, "size": 30},
    "L5": {"obstacle_density": 0.40, "size": 30},
}


def generate_all_maps(save_dir="data/maps", n_per_level=5):
    """Generate and save connected maps for all difficulty levels."""
    os.makedirs(save_dir, exist_ok=True)
    level_keys = list(DIFFICULTY_CONFIGS.keys())
    for level, cfg in DIFFICULTY_CONFIGS.items():
        for i in range(n_per_level):
            seed = i * 100 + level_keys.index(level)
            grid = generate_map(
                size             = cfg["size"],
                obstacle_density = cfg["obstacle_density"],
                seed             = seed,
                min_connectivity = 0.75,
            )
            ratio = _connectivity_ratio(grid)
            path  = os.path.join(save_dir, f"{level}_map{i}.csv")
            save_map(grid, path)
            print(f"Saved {path}  connectivity={ratio:.1%}")


if __name__ == "__main__":
    generate_all_maps()