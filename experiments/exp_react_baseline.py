# exp_react_baseline_v2.py
# Chain-of-Thought (CoT) prompt baseline - simplified ReAct for small models
# The key fix: system prompt FORCES [X,Y] on the FIRST line of response
# so even if the model adds reasoning after, we still get the coordinate.

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np, re
import gymnasium as gym

from stable_baselines3 import PPO
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector
from models.task_memory import TaskMemory
from openai import OpenAI

LLM_URL     = "http://172.20.0.1:1234/v1"
MAP_PATH    = "data/maps/L3_map0.csv"
N_EPISODES  = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ANOMALIES = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
             ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES    = {ANOMALY_NONE:"none", ANOMALY_OBSTRUCTION:"obstruction",
             ANOMALY_DISPLACEMENT:"displacement",
             ANOMALY_INVALIDATION:"invalidation"}

client = OpenAI(base_url=LLM_URL, api_key="lm-studio")

SYSTEM_PROMPT = (
    "You are a robot navigation assistant. "
    "IMPORTANT: Your response MUST start with the coordinate on the FIRST line "
    "in format [X, Y], chosen from the safe candidates list. "
    "You may add brief reasoning AFTER the coordinate. "
    "Example response:\n[22, 15]\nMoving toward the subtarget to recover."
)


def build_ascii_grid(global_map, agent_pos, subtarget, view_radius=8):
    H, W = global_map.shape
    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    r = view_radius
    min_x = max(0, ax-r); max_x = min(H, ax+r+1)
    min_y = max(0, ay-r); max_y = min(W, ay+r+1)
    local = global_map[min_x:max_x, min_y:max_y]
    rel_ax = ax-min_x; rel_ay = ay-min_y
    dx_g = subtarget[0]-ax; dy_g = subtarget[1]-ay

    grid_str = ""; candidates = []
    for i in range(local.shape[0]):
        for j in range(local.shape[1]):
            if i == rel_ax and j == rel_ay:
                grid_str += "R "
            elif local[i,j] == 1:
                grid_str += "X "
            else:
                gi = i+min_x; gj = j+min_y
                dot  = (gi-ax)*dx_g + (gj-ay)*dy_g
                dist = abs(i-rel_ax)+abs(j-rel_ay)
                if dot >= -0.3 and dist >= 3:
                    grid_str += ". "; candidates.append((gi, gj))
                else:
                    grid_str += "- "
        grid_str += "\n"
    candidates.sort(key=lambda p: np.linalg.norm(np.array(p)-np.array(subtarget)))
    top5 = candidates[:5]
    top5_str = ", ".join(f"[{p[0]},{p[1]}]" for p in top5)
    return grid_str, top5, top5_str


def build_cot_prompt(grid_str, dx, dy, top5_str, task_memory, anomaly_type):
    """CoT prompt: coordinate first, then optional reasoning."""
    lines = [
        "Radar map (R=Robot, .=safe candidate, X=wall, -=blocked):",
        grid_str,
        f"Subtarget direction: dx={dx:.0f}, dy={dy:.0f}.",
        "The robot is stuck or needs to recover.",
    ]
    if task_memory is not None:
        lines += ["", "====== Task context ======",
                  task_memory.to_prompt_string(anomaly_type=anomaly_type),
                  "====== End task context ======"]
    lines += [
        "",
        f"Safe candidates: {top5_str}.",
        "",
        "Start your response with the best coordinate [X, Y] on the first line,",
        "then optionally explain your reasoning.",
    ]
    return "\n".join(lines)


def get_cot_waypoint(prompt, top5, agent_pos, global_map):
    try:
        resp = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.01, max_tokens=40
        )
        reply = resp.choices[0].message.content.strip()
        print(f"   [CoT reply] {reply[:80]}")

        # Extract FIRST [X,Y] in response
        m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', reply)
        if m:
            gx, gy = int(m.group(1)), int(m.group(2))
            if (0 <= gx < global_map.shape[0] and
                0 <= gy < global_map.shape[1] and
                global_map[gx, gy] == 0):
                return np.array([gx, gy])
    except Exception as e:
        print(f"   [CoT] exception: {e}")

    return np.array(top5[0]) if top5 else agent_pos.copy()


def run_episode(ppo_model, anomaly_type, seed):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()
    task_memory = TaskMemory(task_type=TASK_PICK_DELIVER)

    pos_history=[]; anomaly_done=False; llm_calls=0
    mode="RL"; waypoint=None; wp_steps=0
    patience=15; stuck_thresh=1.5; safe_dist=5.0; max_wp_steps=25

    for _ in range(2000):
        agent_pos  = env.agent_pos.copy()
        task_phase = env.task_phase
        anomaly    = info["anomaly_type"]
        global_map = info["global_map"]
        subtarget  = info["item_pos"] if task_phase==0 else info["goal_pos"]

        task_memory.update_phase(task_phase)
        if anomaly != ANOMALY_NONE and not anomaly_done:
            task_memory.record_anomaly(anomaly, step=len(pos_history))

        pos_history.append(agent_pos.copy())
        if len(pos_history) > patience: pos_history.pop(0)

        def invoke():
            nonlocal llm_calls, waypoint, mode, wp_steps
            grid_str, top5, top5_str = build_ascii_grid(
                global_map, agent_pos, subtarget)
            dx = subtarget[0]-agent_pos[0]; dy = subtarget[1]-agent_pos[1]
            prompt = build_cot_prompt(grid_str, dx, dy, top5_str,
                                       task_memory, anomaly)
            waypoint = get_cot_waypoint(prompt, top5, agent_pos, global_map)
            llm_calls += 1; mode="LLM"; wp_steps=0

        if anomaly != ANOMALY_NONE and not anomaly_done:
            anomaly_done = True; invoke(); pos_history.clear()

        if mode != "LLM" and (np.linalg.norm(agent_pos-subtarget) >= safe_dist
                               and len(pos_history) == patience):
            if np.linalg.norm(pos_history[-1]-pos_history[0]) < stuck_thresh:
                invoke(); pos_history.clear()

        if mode == "LLM" and waypoint is not None:
            wp_steps += 1
            if np.linalg.norm(agent_pos-waypoint)<=1.5 or wp_steps>max_wp_steps:
                mode="RL"; waypoint=None; pos_history.clear()
                action, _ = ppo_model.predict(obs, deterministic=True)
            else:
                wp_vec = (waypoint-agent_pos).astype(np.float32)
                wp_vec /= (np.linalg.norm(wp_vec)+1e-8)
                fake_obs = np.concatenate([obs[:49], wp_vec, obs[51:]])
                action, _ = ppo_model.predict(fake_obs, deterministic=True)
        else:
            action, _ = ppo_model.predict(obs, deterministic=True)

        obs, _, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc: break

    return {"success": env.task_phase==PHASE_DONE,
            "steps": env.current_step, "llm_calls": llm_calls}


def evaluate():
    print("Loading PPO model...")
    ppo = PPO.load("models_saved/ppo_l3.zip", device="cpu")
    results = {"cot_prompt": {}}

    total_eps = len(ANOMALIES) * N_EPISODES
    done_eps  = 0

    for anomaly in ANOMALIES:
        aname = ANAMES[anomaly]
        print(f"\n{'='*50}\nAnomaly: {aname}\n{'='*50}")
        episodes = []
        for ep in range(N_EPISODES):
            r = run_episode(ppo, anomaly, seed=ep*100)
            episodes.append(r)
            done_eps += 1
            pct  = done_eps / total_eps * 100
            bar  = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
            sr_so_far = np.mean([e["success"] for e in episodes]) * 100
            print(f"  [{bar}] {pct:5.1f}%  "
                  f"ep {ep+1:2d}/{N_EPISODES}  "
                  f"success={r["success"]}  "
                  f"llm={r["llm_calls"]}  "
                  f"SR_so_far={sr_so_far:.0f}%", flush=True)
        sr = np.mean([e['success'] for e in episodes])*100
        results["cot_prompt"][aname] = {
            "success_rate": sr,
            "avg_llm_calls": float(np.mean([e['llm_calls'] for e in episodes]))
        }
        print(f"  → SR={sr:.1f}%")

    path = os.path.join(RESULTS_DIR, "exp_react_baseline.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    # Summary
    try:
        d = json.load(open(os.path.join(RESULTS_DIR,
                                         "exp_llm_prompt_baselines.json")))
        anoms = ['none','obstruction','displacement','invalidation']
        print("\n" + "="*65)
        print(f"{'Strategy':<22} {'None':>8} {'Obstr':>8} "
              f"{'Displ':>8} {'Inval':>8} {'Avg':>8}")
        print("-"*65)
        for name, data in [
            ("Zero-Shot LLM",    d['zero_shot']),
            ("History-Prompt",   d['history_prompt']),
            ("CoT Prompt",       results['cot_prompt']),
            ("Ours (Task Mem.)", d['ours_task_memory']),
        ]:
            srs = [data.get(a,{}).get('success_rate',0) for a in anoms]
            print(f"{name:<22} {srs[0]:>8.1f} {srs[1]:>8.1f} "
                  f"{srs[2]:>8.1f} {srs[3]:>8.1f} {np.mean(srs):>8.1f}")
    except Exception as e:
        print(f"Note: {e}")


if __name__ == "__main__":
    evaluate()