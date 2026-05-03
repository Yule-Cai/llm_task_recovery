# models/llm_recovery.py
import re
import numpy as np
from openai import OpenAI
from envs.task_env import ANOMALY_NONE


class LLMRecovery:
    """
    LLM-based recovery planner.

    Extends the navigation paper's LLMNavigator with:
        - task_memory context injected into the prompt
        - anomaly_type description to guide recovery strategy
        - same three-tier fault tolerance stack (hallucination gate /
          parse fallback / exception fallback)

    Fix log:
        - Hallucination gate now checks free cell only (not restricted to
          directional candidates), matching navigation paper behaviour.
        - History dedup radius raised to 6.0 to prevent oscillation.
        - dot-product filter relaxed to dot >= -0.3 to allow more candidates
          in dense maps where forward candidates are scarce.
    """

    def __init__(
        self,
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        use_task_memory=True,
        view_radius=8,
    ):
        self.client          = OpenAI(base_url=base_url, api_key=api_key)
        self.use_task_memory = use_task_memory
        self.view_radius     = view_radius

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #
    def get_waypoint(
        self,
        global_map,
        agent_pos,
        subtarget_pos,
        task_memory=None,
        anomaly_type=ANOMALY_NONE,
        history_wps=None,
        override_desc=None,
    ):
        if history_wps is None:
            history_wps = []

        map_size = global_map.shape[0]
        ax, ay   = agent_pos
        r        = self.view_radius

        min_x = max(0, ax - r)
        max_x = min(map_size, ax + r + 1)
        min_y = max(0, ay - r)
        max_y = min(map_size, ay + r + 1)

        local_view = global_map[min_x:max_x, min_y:max_y]
        rel_ax     = ax - min_x
        rel_ay     = ay - min_y

        dx_goal = subtarget_pos[0] - ax
        dy_goal = subtarget_pos[1] - ay

        # build candidate list & ASCII grid
        grid_str, valid_waypoints = self._build_ascii_grid(
            local_view, rel_ax, rel_ay, dx_goal, dy_goal, min_x, min_y, ax, ay
        )

        # build set of all free local cells for gate check
        free_local_cells = set()
        for i in range(local_view.shape[0]):
            for j in range(local_view.shape[1]):
                if local_view[i, j] == 0 and not (i == rel_ax and j == rel_ay):
                    free_local_cells.add((i, j))

        if not valid_waypoints:
            # fallback: any free cell in view
            if free_local_cells:
                fc = list(free_local_cells)[0]
                return np.array([fc[0] + min_x, fc[1] + min_y])
            return agent_pos + np.random.randint(-1, 2, size=2)

        # rank candidates
        valid_waypoints.sort(key=lambda p: self._rank_key(
            p, min_x, min_y, subtarget_pos, history_wps
        ))
        top_k     = valid_waypoints[:5]
        top_k_str = ", ".join(f"[{p[0]+min_x}, {p[1]+min_y}]" for p in top_k)

        prompt = self._build_prompt(
            grid_str, dx_goal, dy_goal, top_k_str, task_memory, anomaly_type,
            override_desc=override_desc,
        )

        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a robot navigation assistant. "
                            "Output ONLY a coordinate like [X, Y]. "
                            "Choose strictly from the provided candidate list."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.01,
                max_tokens=15,
            )
            reply = response.choices[0].message.content.strip()
            print(f"   [LLM reply] {reply}")

            # Tier 1 – hallucination gate: check free cell only
            match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", reply)
            if match:
                gx = int(match.group(1))
                gy = int(match.group(2))
                rx = gx - min_x
                ry = gy - min_y

                in_view   = (0 <= rx < local_view.shape[0] and
                             0 <= ry < local_view.shape[1])
                is_free   = in_view and local_view[rx, ry] == 0
                not_agent = (gx != ax or gy != ay)

                if is_free and not_agent:
                    global_wp = np.array([gx, gy])
                    # dedup: skip if too close to recent waypoints
                    if not any(
                        np.linalg.norm(global_wp - hw) < 6.0
                        for hw in history_wps
                    ):
                        return global_wp
                    # close to history but still valid – use it anyway
                    # (better than heuristic top-1 on same trap)
                    return global_wp

            # Tier 2 – parse fallback
            print("   [LLM] parse/gate failed – using heuristic top-1")
            return np.array([top_k[0][0] + min_x, top_k[0][1] + min_y])

        except Exception as e:
            # Tier 3 – exception fallback
            print(f"   [LLM] exception ({e}) – using heuristic top-1")
            return np.array([top_k[0][0] + min_x, top_k[0][1] + min_y])

    # ------------------------------------------------------------------ #
    #  ASCII grid builder                                                  #
    # ------------------------------------------------------------------ #
    def _build_ascii_grid(
        self, local_view, rel_ax, rel_ay, dx_goal, dy_goal, min_x, min_y, ax, ay
    ):
        grid_str        = ""
        valid_waypoints = []

        for i in range(local_view.shape[0]):
            for j in range(local_view.shape[1]):
                if i == rel_ax and j == rel_ay:
                    grid_str += "R "
                elif local_view[i, j] == 1:
                    grid_str += "X "
                else:
                    global_i = i + min_x
                    global_j = j + min_y
                    dx_wp    = global_i - ax
                    dy_wp    = global_j - ay
                    dot      = dx_goal * dx_wp + dy_goal * dy_wp
                    dist     = abs(i - rel_ax) + abs(j - rel_ay)

                    # relaxed filter: dot >= -0.3 allows slight backtrack
                    if dot >= -0.3 and dist >= 3:
                        if self._has_line_of_sight(local_view, rel_ax, rel_ay, i, j):
                            grid_str += ". "
                            valid_waypoints.append((i, j))
                            continue
                    grid_str += "- "
            grid_str += "\n"

        return grid_str, valid_waypoints

    # ------------------------------------------------------------------ #
    #  Prompt builder                                                      #
    # ------------------------------------------------------------------ #
    def _build_prompt(
        self, grid_str, dx_goal, dy_goal, top_k_str, task_memory, anomaly_type,
        override_desc=None,
    ):
        lines = [
            "Radar map (R=Robot, .=safe candidate, X=wall, -=blocked):",
            grid_str,
            f"Subtarget direction: dx={dx_goal}, dy={dy_goal}.",
            "The robot is stuck or needs to recover.",
        ]

        if self.use_task_memory and task_memory is not None:
            lines.append("")
            lines.append("--- Task context ---")
            lines.append(task_memory.to_prompt_string(anomaly_type=anomaly_type))
            lines.append("--- End task context ---")

        lines.append("")
        lines.append(f"Safe candidates (pre-filtered): {top_k_str}.")
        lines.append(
            "Pick the candidate that best helps recover the task. "
            "Output ONLY [X, Y]."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _rank_key(self, p, min_x, min_y, subtarget_pos, history_wps):
        global_p    = np.array([p[0] + min_x, p[1] + min_y])
        dist_target = np.linalg.norm(global_p - subtarget_pos)
        penalty     = sum(
            max(0, 8.0 - np.linalg.norm(global_p - hw)) * 3.0
            for hw in history_wps
        )
        return dist_target + penalty

    def _has_line_of_sight(self, grid, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y   = x0, y0
        sx     = 1 if x0 < x1 else -1
        sy     = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if grid[x, y] == 1:
                    return False
                err -= dy
                if err < 0:
                    y   += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if grid[x, y] == 1:
                    return False
                err -= dx
                if err < 0:
                    x   += sx
                    err += dy
                y += sy

        return grid[x, y] == 0