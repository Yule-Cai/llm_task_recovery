# exp_injection_timing.py
# Tests system robustness across different anomaly injection timings
# Injection at step 30 (early), 100 (default), 200 (mid), 400 (late)
# Run from project root: python exp_injection_timing.py
# Requires LM Studio running with Gemma-3-1B

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from models.arbiter import TaskArbiter
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

LLM_URL    = "http://172.20.0.1:1234/v1"
MAP_PATH   = "data/maps/L3_map0.csv"
N_EPISODES = 20
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Injection timing conditions
TIMINGS  = [30, 100, 200, 400]
ANOMALIES = [ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES    = {ANOMALY_OBSTRUCTION: "obstruction",
             ANOMALY_DISPLACEMENT: "displacement",
             ANOMALY_INVALIDATION: "invalidation"}


def run_episode(ppo_model, anomaly_type, inject_step, seed):
    np.random.seed(seed)
    inj  = AnomalyInjector(anomaly_type=anomaly_type,
                            inject_at_step=inject_step)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    arbiter = TaskArbiter(rl_model=ppo_model, use_task_memory=True,
                          llm_base_url=LLM_URL)
    arbiter.reset(task_type=TASK_PICK_DELIVER)

    for _ in range(2000):
        action, _ = arbiter.predict(obs, info)
        obs, _, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc: break

    return {"success": env.task_phase == PHASE_DONE,
            "steps":   env.current_step,
            "llm_calls": arbiter.llm_calls}


def evaluate():
    print("Loading PPO model...")
    ppo = PPO.load("models_saved/ppo_l3.zip", device="cpu")
    results = {}

    for timing in TIMINGS:
        tkey = f"step_{timing}"
        results[tkey] = {}
        print(f"\n{'='*60}")
        print(f"Injection timing: step {timing}")
        print(f"{'='*60}")

        for anomaly in ANOMALIES:
            aname = ANAMES[anomaly]
            print(f"\n  Anomaly: {aname}")
            episodes = []

            for ep in range(N_EPISODES):
                r = run_episode(ppo, anomaly, timing, seed=ep*100)
                episodes.append(r)
                print(f"    ep {ep+1:2d}  success={r['success']}  "
                      f"steps={r['steps']}  llm={r['llm_calls']}")

            sr = np.mean([e['success'] for e in episodes])*100
            results[tkey][aname] = {
                "success_rate":  sr,
                "avg_llm_calls": float(np.mean([e['llm_calls'] for e in episodes])),
                "inject_step":   timing
            }
            print(f"  → SR={sr:.1f}%")

    path = os.path.join(RESULTS_DIR, "exp_injection_timing.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: SR (%) by injection timing")
    print("="*70)
    print(f"{'Timing':<12} {'Obstruction':>14} {'Displacement':>14} "
          f"{'Invalidation':>14} {'Avg':>8}")
    print("-"*70)
    for tkey in results:
        srs = [results[tkey][a]['success_rate']
               for a in ['obstruction','displacement','invalidation']]
        step = results[tkey]['obstruction']['inject_step']
        print(f"Step {step:<7} {srs[0]:>14.1f} {srs[1]:>14.1f} "
              f"{srs[2]:>14.1f} {np.mean(srs):>8.1f}")


if __name__ == "__main__":
    evaluate()
