# envs/task_env.py
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

TASK_PICK_DELIVER  = 0
TASK_PATROL        = 1
TASK_SEARCH        = 2

TASK_NAMES = {
    TASK_PICK_DELIVER: "pick_deliver",
    TASK_PATROL:       "patrol",
    TASK_SEARCH:       "search_retrieve",
}

PHASE_GO_TO_ITEM   = 0
PHASE_GO_TO_GOAL   = 1
PHASE_GO_TO_WP1    = 0
PHASE_GO_TO_WP2    = 1
PHASE_GO_TO_WP3    = 2
PHASE_EXPLORE      = 0   # Task 3: phase 0 = go to item (far, known location)
PHASE_GO_TO_ITEM_S = 1   # Task 3: phase 1 = go to base after pickup
PHASE_GO_TO_BASE   = 2   # Task 3: phase 2 = done padding (unused)
PHASE_DONE         = 10

ANOMALY_NONE         = "none"
ANOMALY_OBSTRUCTION  = "obstruction"
ANOMALY_DISPLACEMENT = "displacement"
ANOMALY_INVALIDATION = "invalidation"

CELL_FREE = 0.0
CELL_WALL = 1.0
CELL_ITEM = 0.3
CELL_WP   = 0.4
CELL_BASE = 0.5


class TaskEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, map_path=None, max_steps=1200, anomaly_injector=None,
                 radar_radius=3, task_type=None):
        super().__init__()

        if map_path and os.path.exists(map_path):
            self.base_map = pd.read_csv(map_path, header=None).values.astype(np.float32)
        else:
            self.base_map = np.zeros((30, 30), dtype=np.float32)
            self.base_map[0, :] = 1; self.base_map[-1, :] = 1
            self.base_map[:, 0] = 1; self.base_map[:, -1] = 1

        self.map_size         = self.base_map.shape[0]
        self.max_steps        = max_steps
        self.radar_radius     = radar_radius
        self.radar_side       = radar_radius * 2 + 1
        self.anomaly_injector = anomaly_injector
        self._fixed_task      = task_type

        self.action_space = spaces.Discrete(4)
        self._action_delta = {
            0: np.array([-1, 0]), 1: np.array([1, 0]),
            2: np.array([0, -1]), 3: np.array([0, 1]),
        }

        self.observation_space = spaces.Dict({
            "local_grid":    spaces.Box(0.0, 1.0, (self.radar_side, self.radar_side), np.float32),
            "target_vector": spaces.Box(-1.0, 1.0, (2,), np.float32),
            "task_phase":    spaces.Box(0.0, 1.0, (3,), np.float32),
            "task_type":     spaces.Box(0.0, 1.0, (3,), np.float32),
        })

        self.global_map = None; self.agent_pos = None
        self.task_type = TASK_PICK_DELIVER; self.task_phase = 0
        self.current_step = 0; self.anomaly_triggered = False
        self.anomaly_type = ANOMALY_NONE; self.item_picked = False
        self.item_pos = None; self.goal_pos = None
        self.waypoints = []; self.wp_visited = []
        self.item_pos_hidden = None; self.item_visible = False
        self.base_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.global_map = self.base_map.copy()
        self.current_step = 0; self.anomaly_triggered = False
        self.anomaly_type = ANOMALY_NONE; self.item_picked = False
        self.item_visible = False; self.task_phase = 0

        self.task_type = self._fixed_task if self._fixed_task is not None else np.random.randint(0, 3)
        self.agent_pos = self._random_free_pos()

        if self.task_type == TASK_PICK_DELIVER:   self._reset_task1()
        elif self.task_type == TASK_PATROL:        self._reset_task2()
        else:                                      self._reset_task3()

        if self.anomaly_injector is not None:
            self.anomaly_injector.reset(
                global_map=self.global_map, agent_pos=self.agent_pos,
                item_pos=self.item_pos, goal_pos=self.goal_pos,
            )
        return self._get_obs(), self._get_info()

    def _reset_task1(self):
        self.item_pos = self._random_free_pos(exclude=[self.agent_pos], min_dist=6)
        self.goal_pos = self._random_free_pos(exclude=[self.agent_pos, self.item_pos], min_dist=6)

    def _reset_task2(self):
        self.waypoints = []; occupied = [self.agent_pos]
        for _ in range(3):
            wp = self._random_free_pos(exclude=occupied, min_dist=5)
            self.waypoints.append(wp); occupied.append(wp)
        self.wp_visited = [False, False, False]
        self.item_pos = self.waypoints[0]; self.goal_pos = self.waypoints[-1]
        self.base_pos = self.waypoints[-1]

    def _reset_task3(self):
        # Item location is known but far away (min_dist=15)
        # Agent must navigate long distance, making anomaly recovery more critical
        self.item_pos_hidden = self._random_free_pos(exclude=[self.agent_pos], min_dist=10)
        self.base_pos = self._random_free_pos(
            exclude=[self.agent_pos, self.item_pos_hidden], min_dist=6)
        self.item_visible = True  # location known from start
        self.task_phase   = PHASE_EXPLORE  # phase 0 = go to item
        self.item_pos = self.item_pos_hidden; self.goal_pos = self.base_pos

    def step(self, action):
        action = int(action.item()) if hasattr(action, "item") else int(action)
        self.current_step += 1

        if self.anomaly_injector is not None and not self.anomaly_triggered:
            result = self.anomaly_injector.try_inject(
                step=self.current_step, global_map=self.global_map,
                agent_pos=self.agent_pos, item_pos=self.item_pos,
                goal_pos=self.goal_pos, task_phase=self.task_phase,
            )
            if result is not None:
                self.anomaly_triggered = True
                self.anomaly_type = result["type"]
                self._apply_anomaly(result)

        delta = self._action_delta[action]
        new_pos = self.agent_pos + delta
        hit_wall = (new_pos[0] < 0 or new_pos[0] >= self.map_size or
                    new_pos[1] < 0 or new_pos[1] >= self.map_size or
                    self.global_map[new_pos[0], new_pos[1]] == 1)

        reward = 0.0; terminated = False; truncated = False
        if hit_wall:
            reward = -5.0
        else:
            self.agent_pos = new_pos; reward -= 0.1

        if self.task_type == TASK_PICK_DELIVER:
            reward, terminated = self._step_task1(reward, terminated)
        elif self.task_type == TASK_PATROL:
            reward, terminated = self._step_task2(reward, terminated)
        else:
            reward, terminated = self._step_task3(reward, terminated)

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _step_task1(self, reward, terminated):
        if self.task_phase == PHASE_GO_TO_ITEM:
            if np.linalg.norm(self.agent_pos - self.item_pos) <= 1.5:
                self.item_picked = True; self.task_phase = PHASE_GO_TO_GOAL; reward += 50.0
        elif self.task_phase == PHASE_GO_TO_GOAL:
            if np.linalg.norm(self.agent_pos - self.goal_pos) <= 1.5:
                self.task_phase = PHASE_DONE; reward += 200.0; terminated = True
        return reward, terminated

    def _step_task2(self, reward, terminated):
        phase = self.task_phase
        if phase <= 2:
            if np.linalg.norm(self.agent_pos - self.waypoints[phase]) <= 1.5:
                self.wp_visited[phase] = True; reward += 60.0
                if phase == 2:
                    self.task_phase = PHASE_DONE; reward += 100.0; terminated = True
                else:
                    self.task_phase = phase + 1
        return reward, terminated

    def _step_task3(self, reward, terminated):
        if self.task_phase == PHASE_EXPLORE:
            if np.linalg.norm(self.agent_pos - self.item_pos_hidden) <= 1.5:
                self.item_picked = True
                self.item_pos_hidden = self.base_pos.copy()
                self.task_phase = PHASE_GO_TO_ITEM_S
                reward += 50.0
        elif self.task_phase == PHASE_GO_TO_ITEM_S:
            if np.linalg.norm(self.agent_pos - self.base_pos) <= 1.5:
                self.task_phase = PHASE_DONE
                reward += 200.0
                terminated = True
        return reward, terminated

    def _apply_anomaly(self, result):
        atype = result["type"]
        if atype == ANOMALY_DISPLACEMENT:
            new_pos = result.get("new_item_pos")
            if new_pos is not None:
                if self.task_type == TASK_PICK_DELIVER:
                    self.item_pos = new_pos; self.task_phase = PHASE_GO_TO_ITEM
                elif self.task_type == TASK_SEARCH:
                    # item moved but location still known, agent re-navigates
                    self.item_pos_hidden = new_pos; self.item_pos = new_pos
                    self.task_phase = PHASE_EXPLORE  # back to go_to_item
        elif atype == ANOMALY_INVALIDATION:
            new_goal = result.get("new_goal_pos")
            if new_goal is not None:
                self.goal_pos = new_goal
                if self.task_type == TASK_SEARCH:
                    self.base_pos = new_goal

    def _get_obs(self):
        r = self.radar_radius; ax, ay = self.agent_pos
        local_grid = np.ones((self.radar_side, self.radar_side), dtype=np.float32)
        for i in range(self.radar_side):
            for j in range(self.radar_side):
                gx = ax + i - r; gy = ay + j - r
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    local_grid[i, j] = self.global_map[gx, gy]
        self._mark_objects_on_radar(local_grid, ax, ay, r)

        subtarget = self._active_subtarget()
        vec = (subtarget - self.agent_pos).astype(np.float32)
        d = np.linalg.norm(vec)
        if d > 0: vec /= d

        phase_onehot = np.zeros(3, dtype=np.float32)
        if self.task_phase != PHASE_DONE:
            phase_onehot[min(self.task_phase, 2)] = 1.0

        task_onehot = np.zeros(3, dtype=np.float32)
        # Task3 reuses task1 observation encoding so PPO can reuse learned behaviour:
        # task3 phase0 (go_to_item) -> same as task1 phase0
        # task3 phase1 (go_to_base) -> same as task1 phase1 (go_to_goal)
        effective_task = self.task_type
        if self.task_type == TASK_SEARCH:
            effective_task = TASK_PICK_DELIVER
        task_onehot[effective_task] = 1.0

        return {"local_grid": local_grid, "target_vector": vec,
                "task_phase": phase_onehot, "task_type": task_onehot}

    def _mark_objects_on_radar(self, local_grid, ax, ay, r):
        def mark(pos, val):
            if pos is None: return
            px, py = int(pos[0]), int(pos[1])
            li, lj = px - ax + r, py - ay + r
            if 0 <= li < self.radar_side and 0 <= lj < self.radar_side:
                if local_grid[li, lj] == CELL_FREE:
                    local_grid[li, lj] = val

        if self.task_type == TASK_PICK_DELIVER:
            if not self.item_picked: mark(self.item_pos, CELL_ITEM)
        elif self.task_type == TASK_PATROL:
            phase = self.task_phase
            if phase <= 2: mark(self.waypoints[phase], CELL_WP)
        else:
            if self.task_phase == PHASE_EXPLORE:
                mark(self.item_pos_hidden, CELL_ITEM)   # item visible from start
            elif self.task_phase == PHASE_GO_TO_ITEM_S:
                mark(self.base_pos, CELL_BASE)

    def _active_subtarget(self):
        if self.task_type == TASK_PICK_DELIVER:
            return self.item_pos.copy() if self.task_phase == PHASE_GO_TO_ITEM else self.goal_pos.copy()
        elif self.task_type == TASK_PATROL:
            phase = self.task_phase
            return self.waypoints[min(phase, 2)].copy() if phase <= 2 else self.agent_pos.copy()
        else:
            if self.task_phase == PHASE_EXPLORE:
                return self.item_pos_hidden.copy() if self.item_pos_hidden is not None else self.agent_pos.copy()
            elif self.task_phase == PHASE_GO_TO_ITEM_S:
                return self.item_pos_hidden.copy()
            else:
                return self.agent_pos.copy()

    def _get_info(self):
        info = {
            "agent_pos": self.agent_pos.copy(),
            "task_type": self.task_type,
            "task_type_name": TASK_NAMES[self.task_type],
            "task_phase": self.task_phase,
            "anomaly_triggered": self.anomaly_triggered,
            "anomaly_type": self.anomaly_type,
            "global_map": self.global_map,
        }
        if self.task_type == TASK_PICK_DELIVER:
            info.update({"item_pos": self.item_pos.copy(), "goal_pos": self.goal_pos.copy(),
                         "item_picked": self.item_picked})
        elif self.task_type == TASK_PATROL:
            info.update({"waypoints": [wp.copy() for wp in self.waypoints],
                         "wp_visited": self.wp_visited.copy(),
                         "item_pos": self.waypoints[0].copy(),
                         "goal_pos": self.waypoints[-1].copy()})
        else:
            info.update({
                "item_pos":    self.item_pos_hidden.copy() if self.item_pos_hidden is not None else None,
                "goal_pos":    self.base_pos.copy() if self.base_pos is not None else None,
                "item_picked": self.item_picked,
                "item_visible":True,
                "base_pos":    self.base_pos.copy() if self.base_pos is not None else None,
            })
        return info

    def _random_free_pos(self, exclude=None, min_dist=0):
        exclude = exclude or []
        for _ in range(10000):
            x = np.random.randint(1, self.map_size - 1)
            y = np.random.randint(1, self.map_size - 1)
            if self.global_map[x, y] != 0: continue
            pos = np.array([x, y])
            if any(np.linalg.norm(pos - e) < min_dist for e in exclude): continue
            return pos
        raise RuntimeError("Cannot find a free position.")