# eval_lstm.py
# Evaluates LSTM-PPO vs our RL-LLM system on L3
# Tests: all anomaly types, with/without Task Memory context
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from models.arbiter import TaskArbiter
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

LLM_URL    = "http://172.20.0.1:1234/v1"
MAP_PATH   = "data/maps/L3_map0.csv"
N_EPISODES = 30
ANOMALIES  = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
              ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_lstm_episode(lstm_model, anomaly_type):
    """Run one episode with LSTM-PPO (no LLM)."""
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()

    lstm_states = None
    episode_start = True

    for _ in range(2000):
        action, lstm_states = lstm_model.predict(
            obs, state=lstm_states,
            episode_start=np.array([episode_start]),
            deterministic=True)
        episode_start = False
        obs, r, term, trunc, _ = flat.step(action)
        if term or trunc: break

    return {"success": env.task_phase == PHASE_DONE, "steps": env.current_step}


def run_ours_episode(ppo_model, anomaly_type):
    """Run one episode with our RL-LLM + Task Memory."""
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    arbiter = TaskArbiter(rl_model=ppo_model, use_task_memory=True,
                          llm_base_url=LLM_URL)
    arbiter.reset(task_type=TASK_PICK_DELIVER)

    for _ in range(2000):
        action, mode = arbiter.predict(obs, info)
        obs, r, term, trunc, _ = flat.step(action)
        info = env._get_info()
        if term or trunc: break

    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step,
            "llm_calls": arbiter.llm_calls}


def evaluate():
    print("Loading models...")
    lstm_model = RecurrentPPO.load("models_saved/lstm_ppo_l3.zip", device="cpu")
    ppo_model  = PPO.load("models_saved/ppo_l3.zip", device="cpu")
    print("Models loaded.")

    results = {"lstm_ppo": {}, "ours_rl_llm": {}}

    for anomaly in ANOMALIES:
        print("\n" + "="*50)
        print("Anomaly: " + anomaly)
        print("="*50)

        # LSTM-PPO
        print("[LSTM-PPO]")
        lstm_eps = []
        for ep in range(N_EPISODES):
            r = run_lstm_episode(lstm_model, anomaly)
            lstm_eps.append(r)
            print("  ep %2d/%d  success=%s  steps=%d" % (
                ep+1, N_EPISODES, r['success'], r['steps']))

        lstm_sr = np.mean([e['success'] for e in lstm_eps]) * 100
        results["lstm_ppo"][anomaly] = {
            "success_rate": lstm_sr,
            "avg_steps": np.mean([e['steps'] for e in lstm_eps if e['success']]) if any(e['success'] for e in lstm_eps) else None,
            "n_episodes": N_EPISODES
        }
        print("  LSTM-PPO SR=%.1f%%" % lstm_sr)

        # Ours
        print("[Ours (RL+LLM+TaskMemory)]")
        ours_eps = []
        for ep in range(N_EPISODES):
            r = run_ours_episode(ppo_model, anomaly)
            ours_eps.append(r)
            print("  ep %2d/%d  success=%s  steps=%d  llm_calls=%d" % (
                ep+1, N_EPISODES, r['success'], r['steps'], r['llm_calls']))

        ours_sr = np.mean([e['success'] for e in ours_eps]) * 100
        results["ours_rl_llm"][anomaly] = {
            "success_rate": ours_sr,
            "avg_steps": np.mean([e['steps'] for e in ours_eps if e['success']]) if any(e['success'] for e in ours_eps) else None,
            "avg_llm_calls": np.mean([e['llm_calls'] for e in ours_eps]),
            "n_episodes": N_EPISODES
        }
        print("  Ours SR=%.1f%%" % ours_sr)

    # Save
    out = os.path.join(RESULTS_DIR, "exp_lstm_baseline.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to", out)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (L3, pick-and-deliver, 30 episodes)")
    print("="*60)
    print("%-14s %12s %12s %12s %12s" % (
        "Method", "None", "Obstruction", "Displacement", "Invalidation"))
    print("-"*60)
    for method in ["lstm_ppo", "ours_rl_llm"]:
        srs = [str(round(results[method][a]["success_rate"],1))+"%" for a in ANOMALIES]
        print("%-14s %12s %12s %12s %12s" % (method, *srs))


if __name__ == "__main__":
    evaluate()
