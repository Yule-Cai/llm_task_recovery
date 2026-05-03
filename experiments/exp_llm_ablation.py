# exp_llm_ablation.py
# Run once per LLM model:
#   python exp_llm_ablation.py gemma3_1b
#   python exp_llm_ablation.py lfm2.5_1.2b
#   python exp_llm_ablation.py qwen2.5_coder_1.5b
#   python exp_llm_ablation.py qwen3_1.7b

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, time, argparse

from stable_baselines3 import PPO
from models.arbiter import TaskArbiter
from models.llm_recovery import LLMRecovery
from scripts.evaluate import evaluate_agent, get_model_path
from envs.task_env import (ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION,
                            TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH)

LLM_URL    = "http://172.20.0.1:1234/v1"
RESULTS_DIR = "results"
N_EPISODES  = 20
LEVEL       = "L3"
ANOMALIES   = [ANOMALY_NONE, ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
TASKS       = [TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH]
TASK_NAMES  = {TASK_PICK_DELIVER:"pick_deliver", TASK_PATROL:"patrol", TASK_SEARCH:"search_retrieve"}

def make_arbiter(level):
    model = PPO.load(get_model_path(level), device="cpu")
    return TaskArbiter(rl_model=model, use_task_memory=True, llm_base_url=LLM_URL)

def run(model_label):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "exp_llm_" + model_label + ".json")
    results  = {}

    print("\n" + "="*60)
    print("LLM Model: " + model_label)
    print("Make sure this model is loaded in LM Studio at " + LLM_URL)
    print("="*60)

    # Part 1: pick_deliver across all anomaly types (L3)
    print("\n--- Part 1: pick_deliver anomaly types ---")
    results["pick_deliver_anomaly"] = {}
    for anomaly in ANOMALIES:
        print("\nAnomaly: " + anomaly)
        t0 = time.time()
        arbiter = make_arbiter(LEVEL)
        r = evaluate_agent(arbiter=arbiter, anomaly_type=anomaly,
                           level=LEVEL, n_episodes=N_EPISODES,
                           task_type=TASK_PICK_DELIVER)
        r["latency_s"] = round((time.time() - t0) / N_EPISODES, 2)
        results["pick_deliver_anomaly"][anomaly] = r
        print("SR=" + str(r["success_rate"]) + "%  llm_calls=" + str(round(r["avg_llm_calls"],1)))

    # Part 2: multi-task no anomaly (L3)
    print("\n--- Part 2: multi-task no anomaly ---")
    results["multitask_no_anomaly"] = {}
    for task in TASKS:
        name = TASK_NAMES[task]
        print("\nTask: " + name)
        arbiter = make_arbiter(LEVEL)
        r = evaluate_agent(arbiter=arbiter, anomaly_type=ANOMALY_NONE,
                           level=LEVEL, n_episodes=N_EPISODES, task_type=task)
        results["multitask_no_anomaly"][name] = r
        print("SR=" + str(r["success_rate"]) + "%")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to " + out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_label", help="e.g. gemma3_1b / lfm2.5_1.2b / qwen2.5_coder_1.5b / qwen3_1.7b")
    args = parser.parse_args()
    run(args.model_label)
