# eval_lstm_llm.py
# LSTM-PPO + LLM + Task Memory
# Same pipeline as Ours but LSTM replaces PPO as reactive layer
# Run: python eval_lstm_llm.py (needs LM Studio)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from sb3_contrib import RecurrentPPO
from models.llm_recovery import LLMRecovery
from models.task_memory import TaskMemory
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

LLM_URL    = "http://172.20.0.1:1234/v1"
MAP_PATH   = "data/maps/L1_map0.csv"
N_EPISODES = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ANOMALIES = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
             ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES    = {ANOMALY_NONE:"none", ANOMALY_OBSTRUCTION:"obstruction",
             ANOMALY_DISPLACEMENT:"displacement",
             ANOMALY_INVALIDATION:"invalidation"}


def run_episode(lstm_model, llm, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    task_memory  = TaskMemory(task_type=TASK_PICK_DELIVER)
    lstm_states  = None
    ep_start     = True

    pos_history  = []
    anomaly_done = False
    llm_calls    = 0
    mode         = "RL"
    waypoint     = None
    wp_steps     = 0
    patience     = 15
    stuck_thresh = 1.5
    safe_dist    = 5.0
    max_wp_steps = 25

    for _ in range(2000):
        agent_pos  = env.agent_pos.copy()
        task_phase = env.task_phase
        anomaly    = info["anomaly_type"]
        global_map = info["global_map"]
        subtarget  = info["item_pos"] if task_phase == 0 else info["goal_pos"]

        task_memory.update_phase(task_phase)
        if anomaly != ANOMALY_NONE and not anomaly_done:
            task_memory.record_anomaly(anomaly, step=len(pos_history))

        pos_history.append(agent_pos.copy())
        if len(pos_history) > patience:
            pos_history.pop(0)

        def invoke_llm():
            nonlocal llm_calls, waypoint, mode, wp_steps
            wp = llm.get_waypoint(
                global_map=global_map,
                agent_pos=agent_pos,
                subtarget_pos=subtarget,
                task_memory=task_memory,
                anomaly_type=anomaly
            )
            waypoint = wp; llm_calls += 1
            mode = "LLM"; wp_steps = 0

        # Anomaly trigger
        if anomaly != ANOMALY_NONE and not anomaly_done:
            anomaly_done = True
            invoke_llm(); pos_history.clear()

        # Stuck trigger
        if mode != "LLM" and (np.linalg.norm(agent_pos - subtarget) >= safe_dist
                               and len(pos_history) == patience):
            if np.linalg.norm(pos_history[-1] - pos_history[0]) < stuck_thresh:
                invoke_llm(); pos_history.clear()

        # Choose action
        if mode == "LLM" and waypoint is not None:
            wp_steps += 1
            if np.linalg.norm(agent_pos - waypoint) <= 1.5 or wp_steps > max_wp_steps:
                mode = "RL"; waypoint = None; pos_history.clear()
                action, lstm_states = lstm_model.predict(
                    obs, state=lstm_states,
                    episode_start=np.array([ep_start]),
                    deterministic=True)
            else:
                wp_vec = (waypoint - agent_pos).astype(np.float32)
                wp_vec /= (np.linalg.norm(wp_vec) + 1e-8)
                mod_obs = np.concatenate([obs[:49], wp_vec, obs[51:]])
                action, lstm_states = lstm_model.predict(
                    mod_obs, state=lstm_states,
                    episode_start=np.array([ep_start]),
                    deterministic=True)
        else:
            action, lstm_states = lstm_model.predict(
                obs, state=lstm_states,
                episode_start=np.array([ep_start]),
                deterministic=True)

        ep_start = False
        obs, _, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc:
            break

    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step,
            "llm_calls": llm_calls}


def evaluate():
    print("Loading LSTM model...")
    lstm = RecurrentPPO.load("models_saved/lstm_fair_L1.zip", device='cpu')
    llm  = LLMRecovery(base_url=LLM_URL)

    results = {}
    total = len(ANOMALIES) * N_EPISODES
    done  = 0

    for anomaly in ANOMALIES:
        aname = ANAMES[anomaly]
        print(f"\n{'='*50}\nAnomaly: {aname}\n{'='*50}")
        episodes = []

        for ep in range(N_EPISODES):
            r = run_episode(lstm, llm, anomaly, seed=ep * 100)
            episodes.append(r)
            done += 1
            pct = done / total * 100
            bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
            print(f"  [{bar}] {pct:5.1f}%  ep {ep+1:2d}/{N_EPISODES}"
                  f"  success={r['success']}  llm={r['llm_calls']}", flush=True)

        sr = np.mean([e['success'] for e in episodes]) * 100
        avg_calls = float(np.mean([e['llm_calls'] for e in episodes]))
        results[aname] = {"success_rate": sr, "avg_llm_calls": avg_calls}
        print(f"  → SR={sr:.1f}%  avg_llm={avg_calls:.1f}")

    path = os.path.join(RESULTS_DIR, "exp_lstm_llm.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"lstm_llm": results}, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    anoms = ['none','obstruction','displacement','invalidation']
    srs = [results.get(a, {}).get("success_rate", 0) for a in anoms]
    print("\n" + "="*60)
    print("SUMMARY (LSTM+LLM+Task Memory, L1 map, 30 episodes)")
    print("="*60)
    print(f"{'Anomaly':<16} {'SR (%)':>8} {'LLM calls':>12}")
    print("-"*40)
    for a in anoms:
        d = results.get(a, {})
        print(f"{a:<16} {d.get('success_rate',0):>8.1f} "
              f"{d.get('avg_llm_calls',0):>12.1f}")
    print(f"\nAverage SR: {np.mean(srs):.1f}%")


if __name__ == "__main__":
    evaluate()
