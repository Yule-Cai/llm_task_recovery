# exp_search_only.py  — reruns TASK_SEARCH with fixed safe_dist
# The original arbiter uses safe_dist=1.5 for TASK_SEARCH which is too
# aggressive and causes 35-39 LLM calls/episode. This version patches
# the threshold to 8.0 for search, matching the intent of the design.

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from models.arbiter import TaskArbiter
from envs.task_env import (TaskEnv, TASK_SEARCH, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

LLM_URL     = "http://172.20.0.1:1234/v1"
MAP_PATH    = "data/maps/L3_map0.csv"
N_EPISODES  = 20
RESULTS_DIR = "results"

ANOMALIES = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
             ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES    = {ANOMALY_NONE:"none", ANOMALY_OBSTRUCTION:"obstruction",
             ANOMALY_DISPLACEMENT:"displacement",
             ANOMALY_INVALIDATION:"invalidation"}

def run_pure_ppo(ppo_model, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_SEARCH)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    for _ in range(2000):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = flat.step(int(action))
        if term or trunc: break
    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step}

def run_ours(ppo_model, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_SEARCH)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()
    # Fix: use safe_dist=8.0 for search (not 1.5)
    arbiter = TaskArbiter(rl_model=ppo_model, use_task_memory=True,
                          llm_base_url=LLM_URL, safe_dist=8.0)
    arbiter.reset(task_type=TASK_SEARCH)
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

    total_eps = len(ANOMALIES) * N_EPISODES
    done_eps  = 0
    results   = {"search": {}}

    for anomaly in ANOMALIES:
        aname = ANAMES[anomaly]
        print(f"\n{'='*55}\nTask: search  Anomaly: {aname}\n{'='*55}")
        ppo_eps, ours_eps = [], []

        for ep in range(N_EPISODES):
            r_ppo  = run_pure_ppo(ppo, anomaly, seed=ep*100)
            r_ours = run_ours(ppo, anomaly, seed=ep*100)
            ppo_eps.append(r_ppo); ours_eps.append(r_ours)
            done_eps += 1
            pct = done_eps / total_eps * 100
            bar = "#" * int(pct/2) + "-" * (50-int(pct/2))
            print(f"  [{bar}] {pct:5.1f}%  ep {ep+1:2d}/{N_EPISODES}"
                  f"  PPO={r_ppo['success']}  Ours={r_ours['success']}"
                  f"  llm={r_ours['llm_calls']}", flush=True)

        ppo_sr  = np.mean([e['success'] for e in ppo_eps])*100
        ours_sr = np.mean([e['success'] for e in ours_eps])*100
        avg_calls = np.mean([e['llm_calls'] for e in ours_eps])
        results["search"][aname] = {
            "pure_ppo": {"success_rate": ppo_sr},
            "ours":     {"success_rate": ours_sr,
                         "avg_llm_calls": float(avg_calls)}
        }
        print(f"  → PPO={ppo_sr:.1f}%  Ours={ours_sr:.1f}%"
              f"  avg_llm={avg_calls:.1f}")

    # Merge with existing results and save
    path = os.path.join(RESULTS_DIR, "exp_multitask_baseline.json")
    try:
        existing = json.load(open(path, encoding='utf-8'))
    except:
        existing = {}
    existing.update(results)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\nUpdated {path}")

    print("\n" + "="*55)
    print("SEARCH TASK SUMMARY")
    print("="*55)
    print(f"{'Anomaly':<16} {'Pure PPO':>10} {'Ours':>10} {'LLM calls':>12}")
    print("-"*50)
    for aname in ["none","obstruction","displacement","invalidation"]:
        d = results["search"][aname]
        print(f"{aname:<16} {d['pure_ppo']['success_rate']:>10.1f}"
              f" {d['ours']['success_rate']:>10.1f}"
              f" {d['ours']['avg_llm_calls']:>12.1f}")

if __name__ == "__main__":
    evaluate()
