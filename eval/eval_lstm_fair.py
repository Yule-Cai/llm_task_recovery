# eval_lstm_fair.py
# Evaluates the fairly-trained LSTM-PPO baseline
# Run from project root: python eval_lstm_fair.py

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from sb3_contrib import RecurrentPPO
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

MAP_PATH    = "data/maps/L1_map0.csv"
N_EPISODES  = 30
RESULTS_DIR = "results"
ANOMALIES   = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
               ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES      = {ANOMALY_NONE:"none", ANOMALY_OBSTRUCTION:"obstruction",
               ANOMALY_DISPLACEMENT:"displacement",
               ANOMALY_INVALIDATION:"invalidation"}


def run_episode(model, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()

    lstm_states = None
    ep_start    = True
    for _ in range(2000):
        action, lstm_states = model.predict(
            obs, state=lstm_states,
            episode_start=np.array([ep_start]),
            deterministic=True)
        ep_start = False
        obs, _, term, trunc, _ = flat.step(int(action))
        if term or trunc:
            break

    return {"success": env.task_phase == PHASE_DONE,
            "steps":   env.current_step}


def evaluate():
    model_path = "models_saved/lstm_fair_L1.zip"
    print(f"Loading {model_path}...")
    model = RecurrentPPO.load(model_path, device='cpu')

    results = {}
    total = len(ANOMALIES) * N_EPISODES
    done  = 0

    for anomaly in ANOMALIES:
        aname = ANAMES[anomaly]
        print(f"\n  Anomaly: {aname}")
        episodes = []

        for ep in range(N_EPISODES):
            r = run_episode(model, anomaly, seed=ep * 100)
            episodes.append(r)
            done += 1
            pct = done / total * 100
            bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
            print(f"  [{bar}] {pct:5.1f}%  ep {ep+1:2d}/{N_EPISODES}"
                  f"  success={r['success']}", flush=True)

        sr = np.mean([e['success'] for e in episodes]) * 100
        results[aname] = {"success_rate": sr, "n_episodes": N_EPISODES}
        print(f"  → SR={sr:.1f}%")

    # Load our results for comparison
    try:
        d = json.load(open(os.path.join(RESULTS_DIR,
                                         "exp_llm_gemma3_1b.json")))
        ours = {a: d["pick_deliver_anomaly"][a]["success_rate"]
                for a in ["none","obstruction","displacement","invalidation"]}
    except Exception:
        ours = {}

    # Save
    output = {"lstm_fair": results, "ours_llm_task_memory": {
        a: {"success_rate": v} for a, v in ours.items()
    }}
    path = os.path.join(RESULTS_DIR, "exp_lstm_fair_baseline.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    print("\n" + "="*60)
    print("SUMMARY (L1, pick-and-deliver, 30 episodes)")
    print("="*60)
    print(f"{'Method':<28} {'None':>8} {'Obstr':>8} "
          f"{'Displ':>8} {'Inval':>8} {'Avg':>8}")
    print("-"*60)
    anoms = ["none","obstruction","displacement","invalidation"]
    for name, data in [("LSTM-PPO (Fair, 10M steps)", results),
                        ("Ours (LLM+Task Memory)",
                         {a: {"success_rate": v} for a,v in ours.items()})]:
        if not data: continue
        srs = [data.get(a, {}).get("success_rate", 0) for a in anoms]
        print(f"{name:<28} {srs[0]:>8.1f} {srs[1]:>8.1f} "
              f"{srs[2]:>8.1f} {srs[3]:>8.1f} {np.mean(srs):>8.1f}")


if __name__ == "__main__":
    evaluate()