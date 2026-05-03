# models/arbiter.py  — patched for noise intervention experiment
import numpy as np
from models.llm_recovery import LLMRecovery
from models.task_memory import TaskMemory
from envs.task_env import ANOMALY_NONE, TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH


class TaskArbiter:
    def __init__(self, rl_model, use_task_memory=True, patience=15,
                 stuck_threshold=1.5, safe_dist=5.0, trap_radius=8.0,
                 max_waypoint_steps=25, llm_base_url="http://172.20.0.1:1234/v1",
                 override_anomaly_type=None, override_anomaly_desc=None):
        self.rl_model              = rl_model
        self.use_task_memory       = use_task_memory
        self.patience              = patience
        self.stuck_threshold       = stuck_threshold
        self.safe_dist             = safe_dist
        self.trap_radius           = trap_radius
        self.max_waypoint_steps    = max_waypoint_steps
        # Noise intervention: override the anomaly signal sent to LLM
        self.override_anomaly_type = override_anomaly_type
        self.override_anomaly_desc = override_anomaly_desc
        self.llm = LLMRecovery(base_url=llm_base_url, use_task_memory=use_task_memory)
        self.task_memory = None
        self.llm_calls   = 0
        self._reset_state()

    def reset(self, task_type=TASK_PICK_DELIVER):
        self._reset_state()
        self.llm_calls = 0
        if self.use_task_memory:
            self.task_memory = TaskMemory(task_type=task_type)

    def _reset_state(self):
        self.mode              = "RL"
        self.pos_history       = []
        self.current_waypoint  = None
        self.waypoint_steps    = 0
        self.trap_zones        = []
        self.history_wps       = []
        self.used_waypoints    = set()
        self.intervention_logs = []
        self._prev_phase       = -1
        self.anomaly_handled   = False

    def predict(self, obs_flat, info):
        agent_pos    = info["agent_pos"]
        task_type    = info["task_type"]
        task_phase   = info["task_phase"]
        anomaly_type = info["anomaly_type"]
        global_map   = info["global_map"]
        subtarget    = self._get_subtarget(info)

        if self.task_memory is not None:
            self.task_memory.update_phase(task_phase)
            if anomaly_type != ANOMALY_NONE and not self.anomaly_handled:
                self.task_memory.record_anomaly(anomaly_type, step=len(self.pos_history))
        self._prev_phase = task_phase

        radar_flat = obs_flat[:49]
        self.pos_history.append(agent_pos.copy())
        if len(self.pos_history) > self.patience:
            self.pos_history.pop(0)

        if anomaly_type != ANOMALY_NONE and not self.anomaly_handled:
            self.anomaly_handled = True
            # Apply noise override if specified
            llm_anomaly_type = (self.override_anomaly_type
                                if self.override_anomaly_type is not None
                                else anomaly_type)
            print("\n[Arbiter] Anomaly detected: " + anomaly_type +
                  (" -> LLM sees: " + llm_anomaly_type
                   if llm_anomaly_type != anomaly_type else "") +
                  " - calling LLM recovery")
            return self._invoke_llm(radar_flat, agent_pos, subtarget,
                                    global_map, llm_anomaly_type, obs_flat,
                                    override_desc=self.override_anomaly_desc)

        if self.mode == "LLM_WAYPOINT":
            self.waypoint_steps += 1
            dist_wp = np.linalg.norm(agent_pos - self.current_waypoint)
            if dist_wp <= 1.5 or self.waypoint_steps > self.max_waypoint_steps:
                self.mode = "RL"
                self.current_waypoint = None
                self.pos_history.clear()
                return self._rl_with_repulsion(radar_flat, agent_pos, subtarget, obs_flat), "RL"
            wp_vec = (self.current_waypoint - agent_pos).astype(np.float32)
            d = np.linalg.norm(wp_vec)
            if d > 0:
                wp_vec /= d
            fake_obs = np.concatenate([radar_flat, wp_vec, obs_flat[51:]])
            action, _ = self.rl_model.predict(fake_obs, deterministic=True)
            return int(action), "Waypoint"

        dist_to_subtarget = np.linalg.norm(agent_pos - subtarget)
        effective_safe_dist = 1.5 if task_type == TASK_SEARCH else self.safe_dist
        if (dist_to_subtarget >= effective_safe_dist and
                len(self.pos_history) == self.patience):
            displacement = np.linalg.norm(self.pos_history[-1] - self.pos_history[0])
            if displacement < self.stuck_threshold:
                print("\n[Arbiter] Physical stuck detected - calling LLM")
                return self._invoke_llm(radar_flat, agent_pos, subtarget,
                                        global_map, ANOMALY_NONE, obs_flat)

        return self._rl_with_repulsion(radar_flat, agent_pos, subtarget, obs_flat), "RL"

    def _get_subtarget(self, info):
        task_type  = info["task_type"]
        task_phase = info["task_phase"]
        if task_type == TASK_PICK_DELIVER:
            return info["item_pos"] if task_phase == 0 else info["goal_pos"]
        elif task_type == TASK_PATROL:
            wps = info["waypoints"]
            return wps[min(task_phase, len(wps) - 1)]
        else:
            if task_phase == 0:
                m = info["global_map"].shape[0]
                return np.array([m // 2, m // 2], dtype=np.float32)
            elif task_phase == 1:
                return info["item_pos"] if info["item_pos"] is not None else info["agent_pos"]
            else:
                return info["base_pos"] if info.get("base_pos") is not None else info["goal_pos"]

    def _invoke_llm(self, radar_flat, agent_pos, subtarget, global_map,
                    anomaly_type, obs_flat, override_desc=None):
        self.mode = "LLM_WAYPOINT"
        self.waypoint_steps = 0
        self.trap_zones.append(agent_pos.copy())
        self.llm_calls += 1

        new_wp = self.llm.get_waypoint(
            global_map=global_map,
            agent_pos=agent_pos,
            subtarget_pos=subtarget,
            task_memory=self.task_memory,
            anomaly_type=anomaly_type,
            history_wps=self.history_wps,
            override_desc=override_desc,   # pass noisy description to LLM
        )

        wp_key = (int(new_wp[0]), int(new_wp[1]))
        if wp_key in self.used_waypoints:
            offset = np.random.choice([-3, -2, 2, 3], size=2)
            candidate = np.clip(new_wp + offset, 1, global_map.shape[0] - 2)
            if global_map[int(candidate[0]), int(candidate[1])] == 0:
                new_wp = candidate
                wp_key = (int(new_wp[0]), int(new_wp[1]))
        self.used_waypoints.add(wp_key)

        self.current_waypoint = new_wp
        self.history_wps.append(new_wp.copy())
        self.intervention_logs.append({
            "agent_pos":    agent_pos.copy(),
            "waypoint":     new_wp.copy(),
            "anomaly_type": anomaly_type,
        })
        self.pos_history.clear()

        wp_vec = (self.current_waypoint - agent_pos).astype(np.float32)
        d = np.linalg.norm(wp_vec)
        if d > 0:
            wp_vec /= d

        fake_obs = np.concatenate([radar_flat, wp_vec, obs_flat[51:]])
        action, _ = self.rl_model.predict(fake_obs, deterministic=True)
        return int(action), "LLM_invoke"

    def _rl_with_repulsion(self, radar_flat, agent_pos, subtarget, obs_flat):
        target_vec = (subtarget - agent_pos).astype(np.float32)
        if self.trap_zones:
            repulsion = np.zeros(2, dtype=np.float32)
            for trap in self.trap_zones:
                d = np.linalg.norm(agent_pos - trap)
                if d < self.trap_radius:
                    repulsion += (agent_pos - trap) * ((self.trap_radius - d) * 2.0)
            target_vec += repulsion
        d = np.linalg.norm(target_vec)
        if d > 0:
            target_vec /= d
        fake_obs = np.concatenate([radar_flat, target_vec, obs_flat[51:]])
        action, _ = self.rl_model.predict(fake_obs, deterministic=True)
        return int(action)