# exp_trigger_ablation.py
# Compares three trigger mechanisms:
#   1. Ours: physical displacement trigger
#   2. Entropy-based trigger (fire LLM when policy entropy < threshold)
#   3. Fixed-interval trigger (fire LLM every N steps)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, time
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from models.llm_recovery import LLMRecovery
from models.task_memory import TaskMemory
from scripts.evaluate import get_model_path
from envs.task_env import (TaskEnv, ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION,
                            TASK_PICK_DELIVER, PHASE_DONE)
from envs.anomaly_injector import AnomalyInjector

LLM_URL     = "http://172.20.0.1:1234/v1"
RESULTS_DIR = "results"
N_EPISODES  = 30
LEVEL       = "L3"
ANOMALIES   = [ANOMALY_NONE, ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]


# ── Base arbiter with shared RL + LLM logic ───────────────────────────────
class BaseArbiter:
    def __init__(self, rl_model, trap_radius=8.0, max_waypoint_steps=25):
        self.rl_model           = rl_model
        self.trap_radius        = trap_radius
        self.max_waypoint_steps = max_waypoint_steps
        self.llm                = LLMRecovery(base_url=LLM_URL, use_task_memory=True)
        self._reset()

    def reset(self, task_type=TASK_PICK_DELIVER):
        self._reset()
        self.task_memory = TaskMemory(task_type=task_type)

    def _reset(self):
        self.mode             = "RL"
        self.current_waypoint = None
        self.waypoint_steps   = 0
        self.trap_zones       = []
        self.history_wps      = []
        self.used_waypoints   = set()
        self.llm_calls        = 0
        self.task_memory      = None

    def _get_subtarget(self, info):
        task_phase = info["task_phase"]
        return info["item_pos"] if task_phase == 0 else info["goal_pos"]

    def _invoke_llm(self, radar_flat, agent_pos, subtarget, global_map, anomaly_type, obs_flat):
        self.mode = "LLM_WAYPOINT"; self.waypoint_steps = 0
        self.trap_zones.append(agent_pos.copy())
        self.llm_calls += 1

        new_wp = self.llm.get_waypoint(
            global_map=global_map, agent_pos=agent_pos,
            subtarget_pos=subtarget, task_memory=self.task_memory,
            anomaly_type=anomaly_type, history_wps=self.history_wps)

        wp_key = (int(new_wp[0]), int(new_wp[1]))
        if wp_key in self.used_waypoints:
            offset = np.random.choice([-3,-2,2,3], size=2)
            candidate = np.clip(new_wp+offset, 1, global_map.shape[0]-2)
            if global_map[int(candidate[0]), int(candidate[1])] == 0:
                new_wp = candidate; wp_key = (int(new_wp[0]), int(new_wp[1]))
        self.used_waypoints.add(wp_key)
        self.current_waypoint = new_wp
        self.history_wps.append(new_wp.copy())

        wp_vec = (self.current_waypoint - agent_pos).astype(np.float32)
        d = np.linalg.norm(wp_vec)
        if d > 0: wp_vec /= d
        fake_obs = np.concatenate([radar_flat, wp_vec, obs_flat[51:]])
        action, _ = self.rl_model.predict(fake_obs, deterministic=True)
        return int(action), "LLM_invoke"

    def _rl_step(self, radar_flat, agent_pos, subtarget, obs_flat):
        target_vec = (subtarget - agent_pos).astype(np.float32)
        if self.trap_zones:
            rep = np.zeros(2, dtype=np.float32)
            for trap in self.trap_zones:
                d = np.linalg.norm(agent_pos - trap)
                if d < self.trap_radius:
                    rep += (agent_pos - trap) * ((self.trap_radius - d) * 2.0)
            target_vec += rep
        d = np.linalg.norm(target_vec)
        if d > 0: target_vec /= d
        fake_obs = np.concatenate([radar_flat, target_vec, obs_flat[51:]])
        action, _ = self.rl_model.predict(fake_obs, deterministic=True)
        return int(action)

    def _waypoint_tracking(self, radar_flat, agent_pos, subtarget, obs_flat):
        self.waypoint_steps += 1
        dist_wp = np.linalg.norm(agent_pos - self.current_waypoint)
        if dist_wp <= 1.5 or self.waypoint_steps > self.max_waypoint_steps:
            self.mode = "RL"; self.current_waypoint = None
            return self._rl_step(radar_flat, agent_pos, subtarget, obs_flat), "RL"
        wp_vec = (self.current_waypoint - agent_pos).astype(np.float32)
        d = np.linalg.norm(wp_vec)
        if d > 0: wp_vec /= d
        fake_obs = np.concatenate([radar_flat, wp_vec, obs_flat[51:]])
        action, _ = self.rl_model.predict(fake_obs, deterministic=True)
        return int(action), "Waypoint"


# ── 1. Physical displacement trigger (ours) ───────────────────────────────
class DisplacementArbiter(BaseArbiter):
    def __init__(self, rl_model, patience=10, stuck_threshold=2.0, safe_dist=3.0):
        super().__init__(rl_model)
        self.patience         = patience
        self.stuck_threshold  = stuck_threshold
        self.safe_dist        = safe_dist
        self.pos_history      = []
        self.anomaly_handled  = False

    def reset(self, task_type=TASK_PICK_DELIVER):
        super().reset(task_type)
        self.pos_history     = []
        self.anomaly_handled = False

    def predict(self, obs_flat, info):
        agent_pos    = info["agent_pos"]
        anomaly_type = info["anomaly_type"]
        global_map   = info["global_map"]
        subtarget    = self._get_subtarget(info)
        radar_flat   = obs_flat[:49]

        if self.task_memory: self.task_memory.update_phase(info["task_phase"])

        self.pos_history.append(agent_pos.copy())
        if len(self.pos_history) > self.patience: self.pos_history.pop(0)

        if anomaly_type != ANOMALY_NONE and not self.anomaly_handled:
            self.anomaly_handled = True
            return self._invoke_llm(radar_flat, agent_pos, subtarget, global_map, anomaly_type, obs_flat)

        if self.mode == "LLM_WAYPOINT":
            return self._waypoint_tracking(radar_flat, agent_pos, subtarget, obs_flat)

        dist = np.linalg.norm(agent_pos - subtarget)
        if dist >= self.safe_dist and len(self.pos_history) == self.patience:
            disp = np.linalg.norm(self.pos_history[-1] - self.pos_history[0])
            if disp < self.stuck_threshold:
                self.pos_history.clear()
                return self._invoke_llm(radar_flat, agent_pos, subtarget, global_map, ANOMALY_NONE, obs_flat)

        return self._rl_step(radar_flat, agent_pos, subtarget, obs_flat), "RL"


# ── 2. Entropy-based trigger ──────────────────────────────────────────────
class EntropyArbiter(BaseArbiter):
    def __init__(self, rl_model, entropy_threshold=0.3, cooldown=15):
        super().__init__(rl_model)
        self.entropy_threshold = entropy_threshold
        self.cooldown          = cooldown
        self.cooldown_counter  = 0
        self.anomaly_handled   = False

    def reset(self, task_type=TASK_PICK_DELIVER):
        super().reset(task_type)
        self.cooldown_counter = 0
        self.anomaly_handled  = False

    def _get_entropy(self, obs_flat):
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        with torch.no_grad():
            dist = self.rl_model.policy.get_distribution(obs_tensor)
            return dist.entropy().item()

    def predict(self, obs_flat, info):
        agent_pos    = info["agent_pos"]
        anomaly_type = info["anomaly_type"]
        global_map   = info["global_map"]
        subtarget    = self._get_subtarget(info)
        radar_flat   = obs_flat[:49]

        if self.task_memory: self.task_memory.update_phase(info["task_phase"])
        if self.cooldown_counter > 0: self.cooldown_counter -= 1

        if anomaly_type != ANOMALY_NONE and not self.anomaly_handled:
            self.anomaly_handled = True
            return self._invoke_llm(radar_flat, agent_pos, subtarget, global_map, anomaly_type, obs_flat)

        if self.mode == "LLM_WAYPOINT":
            return self._waypoint_tracking(radar_flat, agent_pos, subtarget, obs_flat)

        # entropy trigger
        if self.cooldown_counter == 0:
            entropy = self._get_entropy(obs_flat)
            if entropy < self.entropy_threshold:
                self.cooldown_counter = self.cooldown
                return self._invoke_llm(radar_flat, agent_pos, subtarget, global_map, ANOMALY_NONE, obs_flat)

        return self._rl_step(radar_flat, agent_pos, subtarget, obs_flat), "RL"


# ── 3. Fixed-interval trigger ─────────────────────────────────────────────
class FixedIntervalArbiter(BaseArbiter):
    def __init__(self, rl_model, interval=40):
        super().__init__(rl_model)
        self.interval        = interval
        self.step_counter    = 0
        self.anomaly_handled = False

    def reset(self, task_type=TASK_PICK_DELIVER):
        super().reset(task_type)
        self.step_counter    = 0
        self.anomaly_handled = False

    def predict(self, obs_flat, info):
        agent_pos    = info["agent_pos"]
        anomaly_type = info["anomaly_type"]
        global_map   = info["global_map"]
        subtarget    = self._get_subtarget(info)
        radar_flat   = obs_flat[:49]

        if self.task_memory: self.task_memory.update_phase(info["task_phase"])
        self.step_counter += 1

        if anomaly_type != ANOMALY_NONE and not self.anomaly_handled:
            self.anomaly_handled = True
            return self._invoke_llm(radar_flat, agent_pos, subtarget, global_map, anomaly_type, obs_flat)

        if self.mode == "LLM_WAYPOINT":
            return self._waypoint_tracking(radar_flat, agent_pos, subtarget, obs_flat)

        if self.step_counter % self.interval == 0:
            return self._invoke_llm(radar_flat, agent_pos, subtarget, global_map, ANOMALY_NONE, obs_flat)

        return self._rl_step(radar_flat, agent_pos, subtarget, obs_flat), "RL"


# ── Episode runner ────────────────────────────────────────────────────────
def run_episode(arbiter, map_path, anomaly_type, task_type, max_steps=2000):
    injector = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
                if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=map_path, max_steps=max_steps,
                   anomaly_injector=injector, task_type=task_type)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()
    arbiter.reset(task_type=task_type)

    for _ in range(max_steps):
        action, mode = arbiter.predict(obs, info)
        obs, r, term, trunc, _ = flat.step(action)
        info = env._get_info()
        if term or trunc: break

    return {
        "success":    env.task_phase == PHASE_DONE,
        "steps":      env.current_step,
        "llm_calls":  arbiter.llm_calls,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    map_path  = os.path.join("data", "maps", LEVEL + "_map0.csv")
    ppo_model = PPO.load(get_model_path(LEVEL), device="cpu")

    arbiters = {
        "displacement": DisplacementArbiter(PPO.load(get_model_path(LEVEL), device="cpu")),
        "entropy":      EntropyArbiter(PPO.load(get_model_path(LEVEL), device="cpu")),
        "fixed_40":     FixedIntervalArbiter(PPO.load(get_model_path(LEVEL), device="cpu"), interval=40),
    }

    results = {name: {} for name in arbiters}

    for anomaly in ANOMALIES:
        print("\n" + "="*55)
        print("Anomaly: " + anomaly)
        print("="*55)
        for name, arbiter in arbiters.items():
            episodes = []
            for ep in range(N_EPISODES):
                r = run_episode(arbiter, map_path, anomaly, TASK_PICK_DELIVER)
                episodes.append(r)

            sr        = np.mean([e["success"] for e in episodes]) * 100
            avg_calls = np.mean([e["llm_calls"] for e in episodes])
            avg_steps = np.mean([e["steps"] for e in episodes if e["success"]]) if any(e["success"] for e in episodes) else None

            results[name][anomaly] = {
                "success_rate":  sr,
                "avg_llm_calls": avg_calls,
                "avg_steps":     avg_steps,
            }
            print(f"  {name:<16} SR={sr:.0f}%  llm_calls={avg_calls:.1f}")

    out_path = os.path.join(RESULTS_DIR, "exp_trigger_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to " + out_path)


if __name__ == "__main__":
    main()
