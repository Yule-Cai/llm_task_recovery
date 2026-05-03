# envs/anomaly_injector.py
import numpy as np
from envs.task_env import ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION, PHASE_GO_TO_ITEM


class AnomalyInjector:
    """
    Injects one of three anomaly types at a specified step during an episode.

    Parameters
    ----------
    anomaly_type : str
        One of "obstruction", "displacement", "invalidation", "none".
    inject_at_step : int or None
        Step at which to trigger the anomaly.
        If None, triggers at a random step between inject_range.
    inject_range : tuple
        (min_step, max_step) for random injection timing.
    n_obstruction_blocks : int
        Number of wall cells to place for obstruction anomaly.
    """

    def __init__(
        self,
        anomaly_type="obstruction",
        inject_at_step=None,
        inject_range=(50, 150),
        n_obstruction_blocks=5,
    ):
        self.anomaly_type          = anomaly_type
        self.inject_at_step        = inject_at_step
        self.inject_range          = inject_range
        self.n_obstruction_blocks  = n_obstruction_blocks

        # set during reset()
        self._trigger_step  = None
        self._global_map    = None
        self._agent_pos     = None
        self._item_pos      = None
        self._goal_pos      = None

    # ------------------------------------------------------------------ #
    #  Episode initialisation                                              #
    # ------------------------------------------------------------------ #
    def reset(self, global_map, agent_pos, item_pos, goal_pos):
        self._global_map = global_map   # reference – modifications are in-place
        self._agent_pos  = agent_pos
        self._item_pos   = item_pos
        self._goal_pos   = goal_pos

        if self.inject_at_step is not None:
            self._trigger_step = self.inject_at_step
        else:
            self._trigger_step = np.random.randint(
                self.inject_range[0], self.inject_range[1]
            )

    # ------------------------------------------------------------------ #
    #  Called every step from TaskEnv.step()                              #
    # ------------------------------------------------------------------ #
    def try_inject(self, step, global_map, agent_pos, item_pos, goal_pos, task_phase):
        """
        Returns a result dict if anomaly fires this step, else None.

        Result dict always contains key "type".
        Additional keys depend on anomaly type:
            obstruction  – no extra keys (map modified in-place)
            displacement – "new_item_pos"
            invalidation – "new_goal_pos"
        """
        if self.anomaly_type == "none":
            return None
        if step != self._trigger_step:
            return None

        if self.anomaly_type == ANOMALY_OBSTRUCTION:
            return self._inject_obstruction(global_map, agent_pos, item_pos, goal_pos)

        if self.anomaly_type == ANOMALY_DISPLACEMENT:
            return self._inject_displacement(global_map, agent_pos, goal_pos)

        if self.anomaly_type == ANOMALY_INVALIDATION:
            return self._inject_invalidation(global_map, agent_pos, item_pos, goal_pos)

        return None

    # ------------------------------------------------------------------ #
    #  Anomaly implementations                                            #
    # ------------------------------------------------------------------ #
    def _inject_obstruction(self, global_map, agent_pos, item_pos, goal_pos):
        """Place wall blocks between agent and current subtarget."""
        map_size   = global_map.shape[0]
        placed     = 0
        attempts   = 0

        # midpoint between agent and item as seed for block placement
        mid = ((agent_pos + item_pos) // 2).astype(int)

        while placed < self.n_obstruction_blocks and attempts < 500:
            attempts += 1
            offset = np.random.randint(-4, 5, size=2)
            pos    = mid + offset
            px, py = int(pos[0]), int(pos[1])

            if not (1 <= px < map_size - 1 and 1 <= py < map_size - 1):
                continue
            if global_map[px, py] == 1:
                continue
            # do not block agent, item, or goal cells
            if np.array_equal([px, py], agent_pos): continue
            if np.array_equal([px, py], item_pos):  continue
            if np.array_equal([px, py], goal_pos):  continue

            global_map[px, py] = 1
            placed += 1

        return {"type": ANOMALY_OBSTRUCTION}

    def _inject_displacement(self, global_map, agent_pos, goal_pos):
        """Move item to a new random free location."""
        map_size = global_map.shape[0]
        for _ in range(10000):
            x = np.random.randint(1, map_size - 1)
            y = np.random.randint(1, map_size - 1)
            if global_map[x, y] != 0:
                continue
            new_pos = np.array([x, y])
            if np.linalg.norm(new_pos - agent_pos) < 5:
                continue
            if np.linalg.norm(new_pos - goal_pos) < 3:
                continue
            return {"type": ANOMALY_DISPLACEMENT, "new_item_pos": new_pos}

        # fallback: return original position unchanged (no crash)
        return {"type": ANOMALY_DISPLACEMENT, "new_item_pos": self._item_pos.copy()}

    def _inject_invalidation(self, global_map, agent_pos, item_pos, goal_pos):
        """Block the original goal zone and provide an alternate goal."""
        map_size = global_map.shape[0]

        # wall off a small region around original goal
        gx, gy = int(goal_pos[0]), int(goal_pos[1])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if not (0 <= nx < map_size and 0 <= ny < map_size):
                    continue
                if np.array_equal([nx, ny], agent_pos): continue
                if np.array_equal([nx, ny], item_pos):  continue
                global_map[nx, ny] = 1

        # find new goal far from original
        for _ in range(10000):
            x = np.random.randint(1, map_size - 1)
            y = np.random.randint(1, map_size - 1)
            if global_map[x, y] != 0:
                continue
            new_goal = np.array([x, y])
            if np.linalg.norm(new_goal - goal_pos)  < 8:  continue
            if np.linalg.norm(new_goal - agent_pos) < 5:  continue
            return {"type": ANOMALY_INVALIDATION, "new_goal_pos": new_goal}

        return {"type": ANOMALY_INVALIDATION, "new_goal_pos": goal_pos.copy()}
