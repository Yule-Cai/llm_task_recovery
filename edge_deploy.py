# edge_deploy.py
# Run on Orange Pi 4 via SSH
# Tests the full RL-LLM task recovery system on edge hardware

import os
import sys
import time
import json
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT,
                            ANOMALY_INVALIDATION, PHASE_DONE)
from envs.anomaly_injector import AnomalyInjector
from models.arbiter import TaskArbiter

# ── Config ────────────────────────────────────────────────────────────────
LLM_URL    = "http://localhost:1234/v1"
MODEL_PATH = "ppo_l3.zip"
MAP_PATH   = "L5_map0.csv"
N_EPISODES = 10
MAX_STEPS  = 2000
RESULTS_FILE = "edge_results.json"

TEST_CONFIGS = [
    {"task": TASK_PICK_DELIVER, "anomaly": ANOMALY_NONE,        "label": "pick_deliver_none"},
    {"task": TASK_PICK_DELIVER, "anomaly": ANOMALY_OBSTRUCTION, "label": "pick_deliver_obstruction"},
    {"task": TASK_PICK_DELIVER, "anomaly": ANOMALY_DISPLACEMENT,"label": "pick_deliver_displacement"},
    {"task": TASK_PICK_DELIVER, "anomaly": ANOMALY_INVALIDATION,"label": "pick_deliver_invalidation"},
    {"task": TASK_PATROL,       "anomaly": ANOMALY_NONE,        "label": "patrol_none"},
    {"task": TASK_PATROL,       "anomaly": ANOMALY_OBSTRUCTION, "label": "patrol_obstruction"},
    {"task": TASK_SEARCH,       "anomaly": ANOMALY_NONE,        "label": "search_none"},
    {"task": TASK_SEARCH,       "anomaly": ANOMALY_OBSTRUCTION, "label": "search_obstruction"},
    {"task": TASK_SEARCH,       "anomaly": ANOMALY_DISPLACEMENT,"label": "search_displacement"},
]


def run_episode(ppo_model, task_type, anomaly_type):
    injector = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
                if anomaly_type != ANOMALY_NONE else None)

    env  = TaskEnv(map_path=MAP_PATH, max_steps=MAX_STEPS,
                   anomaly_injector=injector, task_type=task_type)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    arbiter = TaskArbiter(rl_model=ppo_model, use_task_memory=True,
                          llm_base_url=LLM_URL)
    arbiter.reset(task_type=task_type)

    llm_latencies = []
    original_get_wp = arbiter.llm.get_waypoint

    def timed_get_wp(*args, **kwargs):
        t0  = time.time()
        res = original_get_wp(*args, **kwargs)
        llm_latencies.append(time.time() - t0)
        return res

    arbiter.llm.get_waypoint = timed_get_wp

    for _ in range(MAX_STEPS):
        action, mode = arbiter.predict(obs, info)
        obs, r, term, trunc, _ = flat.step(action)
        info = env._get_info()
        if term or trunc: break

    success = (env.task_phase == PHASE_DONE)
    return {
        "success":       success,
        "steps":         env.current_step,
        "llm_calls":     len(llm_latencies),
        "mean_latency":  round(float(np.mean(llm_latencies)), 2) if llm_latencies else 0,
        "peak_latency":  round(float(np.max(llm_latencies)), 2)  if llm_latencies else 0,
    }


def main():
    print("="*55)
    print("Edge Deployment Test — Orange Pi 4")
    print("LLM endpoint:", LLM_URL)
    print("Map:", MAP_PATH)
    print("="*55)

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found:", MODEL_PATH)
        sys.exit(1)
    if not os.path.exists(MAP_PATH):
        print("ERROR: Map not found:", MAP_PATH)
        sys.exit(1)

    print("\nLoading PPO model...")
    ppo_model = PPO.load(MODEL_PATH, device="cpu")
    print("Model loaded.")

    all_results = {}
    grand_start = time.time()

    for cfg in TEST_CONFIGS:
        label      = cfg["label"]
        task_type  = cfg["task"]
        anomaly    = cfg["anomaly"]

        print("\n" + "-"*50)
        print("Config:", label)
        episodes = []

        for ep in range(N_EPISODES):
            ep_start = time.time()
            result   = run_episode(ppo_model, task_type, anomaly)
            elapsed  = round(time.time() - ep_start, 1)
            episodes.append(result)
            print(f"  ep {ep+1:2d}/{N_EPISODES}  "
                  f"success={result['success']}  "
                  f"steps={result['steps']}  "
                  f"llm_calls={result['llm_calls']}  "
                  f"mean_lat={result['mean_latency']}s  "
                  f"ep_time={elapsed}s")

        successes    = [e["success"]      for e in episodes]
        all_latencies= [l for e in episodes
                        for l in ([e["mean_latency"]] if e["llm_calls"] > 0 else [])]

        summary = {
            "success_rate":     round(np.mean(successes) * 100, 1),
            "avg_steps":        round(float(np.mean([e["steps"] for e in episodes if e["success"]])), 1)
                                if any(successes) else None,
            "avg_llm_calls":    round(float(np.mean([e["llm_calls"] for e in episodes])), 2),
            "mean_latency_s":   round(float(np.mean(all_latencies)), 2) if all_latencies else 0,
            "peak_latency_s":   round(float(np.max([e["peak_latency"] for e in episodes])), 2),
            "n_episodes":       N_EPISODES,
            "raw":              episodes,
        }
        all_results[label] = summary
        print(f"  SR={summary['success_rate']}%  "
              f"mean_lat={summary['mean_latency_s']}s  "
              f"peak_lat={summary['peak_latency_s']}s")

    total_time = round(time.time() - grand_start, 1)
    all_results["_meta"] = {
        "device":     "Orange Pi 4",
        "llm":        "Gemma-3-1B via LM Studio",
        "total_time": total_time,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*55)
    print("All tests complete in", total_time, "seconds")
    print("Results saved to:", RESULTS_FILE)
    print("="*55)

    print("\nSummary:")
    print(f"{'Config':<30} {'SR':>6} {'Mean Lat':>10} {'Peak Lat':>10}")
    print("-"*58)
    for label, r in all_results.items():
        if label.startswith("_"): continue
        sr   = str(r["success_rate"]) + "%"
        mlat = str(r["mean_latency_s"]) + "s"
        plat = str(r["peak_latency_s"]) + "s"
        print(f"{label:<30} {sr:>6} {mlat:>10} {plat:>10}")


if __name__ == "__main__":
    main()
