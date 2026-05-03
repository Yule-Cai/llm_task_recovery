# scripts/evaluate.py
import os, sys, numpy as np, gymnasium as gym

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.task_env import (TaskEnv, ANOMALY_NONE, PHASE_DONE,
                            TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH)
from envs.anomaly_injector import AnomalyInjector

MAP_DIR        = os.path.join(project_root, "data", "maps")
MODEL_SAVE_DIR = os.path.join(project_root, "models_saved")
LLM_BASE_URL   = "http://172.20.0.1:1234/v1"


def get_model_path(level):
    candidates = {
        "L1": ["ppo_l3", "ppo_l2", "ppo_l1"],
        "L2": ["ppo_l3", "ppo_l2", "ppo_l1"],
        "L3": ["ppo_l3"],
        "L4": ["ppo_l4", "ppo_l3"],
        "L5": ["ppo_l5", "ppo_l4", "ppo_l3"],
    }
    for name in candidates.get(level, ["ppo_l3"]):
        path = os.path.join(MODEL_SAVE_DIR, name + ".zip")
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No model found for level " + level)


def run_episode(env_raw, arbiter, max_steps=1200):
    from envs.task_env import TASK_PICK_DELIVER
    flat_env = gym.wrappers.FlattenObservation(env_raw)
    obs, _   = flat_env.reset()
    info     = env_raw._get_info()

    task_type = info["task_type"]
    if arbiter is not None:
        arbiter.reset(task_type=task_type)

    llm_calls = 0; total_steps = 0

    for _ in range(max_steps):
        total_steps += 1
        if arbiter is not None:
            action, mode = arbiter.predict(obs, info)
            if mode == "LLM_invoke":
                llm_calls += 1
        else:
            action, _ = env_raw._pure_ppo_model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, _ = flat_env.step(action)
        info = env_raw._get_info()
        if terminated or truncated:
            break

    success = (env_raw.task_phase == PHASE_DONE)
    return {"success": success, "steps": total_steps,
            "llm_calls": llm_calls, "anomaly_type": env_raw.anomaly_type,
            "task_type": env_raw.task_type}


def evaluate_agent(arbiter, anomaly_type=ANOMALY_NONE, level="L3",
                   n_episodes=10, map_idx=0, inject_at_step=100,
                   pure_ppo_model=None, task_type=TASK_PICK_DELIVER):
    from stable_baselines3 import PPO
    from models.arbiter import TaskArbiter

    map_path = os.path.join(MAP_DIR, level + "_map" + str(map_idx) + ".csv")

    if arbiter is not None and hasattr(arbiter, 'rl_model'):
        model_path = get_model_path(level)
        arbiter.rl_model = PPO.load(model_path, device="cpu")

    results = []
    for ep in range(n_episodes):
        injector = (AnomalyInjector(anomaly_type=anomaly_type,
                                    inject_at_step=inject_at_step)
                    if anomaly_type != ANOMALY_NONE else None)

        env = TaskEnv(map_path=map_path, max_steps=1200,
                      anomaly_injector=injector, task_type=task_type)

        if pure_ppo_model is not None:
            env._pure_ppo_model = pure_ppo_model

        result = run_episode(env, arbiter=arbiter)
        results.append(result)
        print("  ep %2d/%d  success=%s  steps=%d  llm_calls=%d" % (
            ep+1, n_episodes, result['success'], result['steps'], result['llm_calls']))

    successes  = [r["success"] for r in results]
    steps_succ = [r["steps"]   for r in results if r["success"]]
    return {
        "success_rate":  np.mean(successes) * 100,
        "avg_steps":     np.mean(steps_succ) if steps_succ else None,
        "avg_llm_calls": np.mean([r["llm_calls"] for r in results]),
        "n_episodes":    n_episodes,
        "raw":           results,
    }