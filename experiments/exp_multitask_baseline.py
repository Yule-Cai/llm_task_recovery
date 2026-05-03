# exp_multitask_baseline.py
# Extends baseline comparison (Pure PPO vs Rule-based vs Ours) to all 3 task types
# Run from project root: python exp_multitask_baseline.py
# Requires LM Studio running with Gemma-3-1B

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from models.arbiter import TaskArbiter
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, TASK_PATROL,
                            TASK_SEARCH, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

LLM_URL    = "http://172.20.0.1:1234/v1"
MAP_PATH   = "data/maps/L3_map0.csv"
N_EPISODES = 20
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASKS    = [TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH]
TNAMES   = {TASK_PICK_DELIVER: "pick_deliver",
            TASK_PATROL:       "patrol",
            TASK_SEARCH:       "search"}
ANOMALIES = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
             ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES    = {ANOMALY_NONE: "none", ANOMALY_OBSTRUCTION: "obstruction",
             ANOMALY_DISPLACEMENT: "displacement",
             ANOMALY_INVALIDATION: "invalidation"}


def run_pure_ppo(ppo_model, task_type, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=task_type)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    for _ in range(2000):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = flat.step(int(action))
        if term or trunc: break
    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step}


def run_ours(ppo_model, task_type, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=task_type)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()
    arbiter = TaskArbiter(rl_model=ppo_model, use_task_memory=True,
                          llm_base_url=LLM_URL)
    arbiter.reset(task_type=task_type)
    for _ in range(2000):
        action, _ = arbiter.predict(obs, info)
        obs, _, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc: break
    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step,
            "llm_calls": arbiter.llm_calls}


def evaluate():
    print("Loading PPO model...")
    ppo = PPO.load("models_saved/ppo_l3.zip", device="cpu")
    results = {}

    for task in TASKS:
        tname = TNAMES[task]
        results[tname] = {}
        print(f"\n{'='*60}\nTask: {tname}\n{'='*60}")

        for anomaly in ANOMALIES:
            aname = ANAMES[anomaly]
            print(f"\n  Anomaly: {aname}")
            ppo_eps, ours_eps = [], []

            for ep in range(N_EPISODES):
                r_ppo  = run_pure_ppo(ppo, task, anomaly, seed=ep*100)
                r_ours = run_ours(ppo, task, anomaly, seed=ep*100)
                ppo_eps.append(r_ppo)
                ours_eps.append(r_ours)
                print(f"    ep {ep+1:2d}  PPO={r_ppo['success']}  "
                      f"Ours={r_ours['success']}")

            results[tname][aname] = {
                "pure_ppo": {
                    "success_rate": np.mean([e['success'] for e in ppo_eps])*100
                },
                "ours": {
                    "success_rate": np.mean([e['success'] for e in ours_eps])*100,
                    "avg_llm_calls": np.mean([e['llm_calls'] for e in ours_eps])
                }
            }
            print(f"  → PPO={results[tname][aname]['pure_ppo']['success_rate']:.1f}%  "
                  f"Ours={results[tname][aname]['ours']['success_rate']:.1f}%")

    path = os.path.join(RESULTS_DIR, "exp_multitask_baseline.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for tname in results:
        print(f"\n{tname}:")
        print(f"  {'Anomaly':<16} {'Pure PPO':>10} {'Ours':>10}")
        print(f"  {'-'*38}")
        for aname in results[tname]:
            psr = results[tname][aname]['pure_ppo']['success_rate']
            osr = results[tname][aname]['ours']['success_rate']
            print(f"  {aname:<16} {psr:>10.1f} {osr:>10.1f}")


if __name__ == "__main__":
    evaluate()
