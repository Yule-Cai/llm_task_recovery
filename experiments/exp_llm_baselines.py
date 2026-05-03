# exp_llm_baselines.py
# Compares three LLM prompt strategies as cognitive baselines:
#   1. Ours (Task Memory + anomaly description) - from existing results
#   2. Zero-Shot LLM (current position + target only, no task context)
#   3. History-Prompt LLM (last N trajectory coords instead of Task Memory)
#
# Run from project root: python exp_llm_baselines.py
# Requires LM Studio running with Gemma-3-1B at http://172.20.0.1:1234/v1

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector
from models.task_memory import TaskMemory
from openai import OpenAI
import re

LLM_URL    = "http://172.20.0.1:1234/v1"
MAP_PATH   = "data/maps/L3_map0.csv"
N_EPISODES = 30
ANOMALIES  = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
              ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES     = {ANOMALY_NONE: "none", ANOMALY_OBSTRUCTION: "obstruction",
              ANOMALY_DISPLACEMENT: "displacement",
              ANOMALY_INVALIDATION: "invalidation"}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

client = OpenAI(base_url=LLM_URL, api_key="lm-studio")


# ── Shared helpers ─────────────────────────────────────────────────────────
def get_llm_waypoint_from_prompt(prompt):
    """Call LLM and parse [x, y] waypoint with three-tier fallback."""
    try:
        resp = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system",
                 "content": ("You are a robot navigation assistant. "
                             "Output ONLY a coordinate like [X, Y]. "
                             "Choose strictly from the provided candidate list.")},
                {"role": "user", "content": prompt}
            ],
            temperature=0.01, max_tokens=15
        )
        reply = resp.choices[0].message.content.strip()
        m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", reply)
        if m:
            return int(m.group(1)), int(m.group(2)), True
    except Exception:
        pass
    return None, None, False


def build_ascii_grid(global_map, agent_pos, subtarget_pos, view_radius=8):
    """Build ASCII grid and candidate waypoints (same as LLMRecovery)."""
    H, W = global_map.shape
    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    r = view_radius
    min_x = max(0, ax - r); max_x = min(H, ax + r + 1)
    min_y = max(0, ay - r); max_y = min(W, ay + r + 1)
    local = global_map[min_x:max_x, min_y:max_y]
    rel_ax = ax - min_x; rel_ay = ay - min_y
    dx_g = subtarget_pos[0] - ax; dy_g = subtarget_pos[1] - ay

    grid_str = ""
    candidates = []
    for i in range(local.shape[0]):
        for j in range(local.shape[1]):
            if i == rel_ax and j == rel_ay:
                grid_str += "R "
            elif local[i, j] == 1:
                grid_str += "X "
            else:
                gi = i + min_x; gj = j + min_y
                dx_wp = gi - ax; dy_wp = gj - ay
                dot  = dx_g * dx_wp + dy_g * dy_wp
                dist = abs(i - rel_ax) + abs(j - rel_ay)
                if dot >= -0.3 and dist >= 3:
                    grid_str += ". "; candidates.append((gi, gj))
                else:
                    grid_str += "- "
        grid_str += "\n"

    candidates.sort(key=lambda p: np.linalg.norm(
        np.array(p) - np.array(subtarget_pos)))
    top5 = candidates[:5]
    top5_str = ", ".join(f"[{p[0]}, {p[1]}]" for p in top5)
    return grid_str, top5, top5_str, (min_x, min_y)


def heuristic_fallback(top5, agent_pos, global_map):
    """Return best free candidate toward subtarget."""
    if top5:
        return np.array(top5[0])
    return agent_pos.copy()


# ── Prompt builders ────────────────────────────────────────────────────────
def build_zeroshot_prompt(grid_str, dx_goal, dy_goal, top5_str):
    """Zero-Shot: only map + current direction, no task context."""
    return "\n".join([
        "Radar map (R=Robot, .=safe candidate, X=wall, -=blocked):",
        grid_str,
        f"Target direction: dx={dx_goal:.0f}, dy={dy_goal:.0f}.",
        "The robot is stuck or needs to recover.",
        "",
        f"Safe candidates: {top5_str}.",
        "Pick the best candidate toward the target. Output ONLY [X, Y]."
    ])


def build_history_prompt(grid_str, dx_goal, dy_goal, top5_str,
                         traj_history, anomaly_type):
    """History-Prompt: last N trajectory coords instead of Task Memory."""
    hist_str = " -> ".join(f"[{p[0]},{p[1]}]" for p in traj_history[-10:])
    anom_str = ""
    if anomaly_type != ANOMALY_NONE:
        anom_str = f"\nAnomaly detected: {anomaly_type}."
    return "\n".join([
        "Radar map (R=Robot, .=safe candidate, X=wall, -=blocked):",
        grid_str,
        f"Target direction: dx={dx_goal:.0f}, dy={dy_goal:.0f}.",
        f"Recent trajectory (last 10 steps): {hist_str}",
        "The robot is stuck or needs to recover." + anom_str,
        "",
        f"Safe candidates: {top5_str}.",
        "Pick the best candidate to recover toward the target. "
        "Output ONLY [X, Y]."
    ])


# ── Episode runners ────────────────────────────────────────────────────────
def run_episode_zeroshot(ppo_model, anomaly_type, seed):
    """Run one episode with Zero-Shot LLM (no task context)."""
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    pos_history   = []
    anomaly_done  = False
    llm_calls     = 0
    mode          = "RL"
    waypoint      = None
    wp_steps      = 0
    patience      = 15
    stuck_thresh  = 1.5
    safe_dist     = 5.0
    max_wp_steps  = 25

    for step in range(2000):
        agent_pos = env.agent_pos.copy()
        task_phase = env.task_phase
        anomaly    = info["anomaly_type"]
        global_map = info["global_map"]
        subtarget  = (info["item_pos"] if task_phase == 0
                      else info["goal_pos"])

        pos_history.append(agent_pos.copy())
        if len(pos_history) > patience:
            pos_history.pop(0)

        # Anomaly trigger
        if anomaly != ANOMALY_NONE and not anomaly_done:
            anomaly_done = True
            grid_str, top5, top5_str, _ = build_ascii_grid(
                global_map, agent_pos, subtarget)
            dx = subtarget[0] - agent_pos[0]
            dy = subtarget[1] - agent_pos[1]
            prompt = build_zeroshot_prompt(grid_str, dx, dy, top5_str)
            gx, gy, ok = get_llm_waypoint_from_prompt(prompt)
            llm_calls += 1
            if ok and global_map[gx, gy] == 0:
                waypoint = np.array([gx, gy])
            else:
                waypoint = heuristic_fallback(top5, agent_pos, global_map)
            mode = "LLM"; wp_steps = 0

        # Displacement trigger
        if mode != "LLM" and (np.linalg.norm(agent_pos - subtarget) >= safe_dist
                               and len(pos_history) == patience):
            disp = np.linalg.norm(pos_history[-1] - pos_history[0])
            if disp < stuck_thresh:
                grid_str, top5, top5_str, _ = build_ascii_grid(
                    global_map, agent_pos, subtarget)
                dx = subtarget[0] - agent_pos[0]
                dy = subtarget[1] - agent_pos[1]
                prompt = build_zeroshot_prompt(grid_str, dx, dy, top5_str)
                gx, gy, ok = get_llm_waypoint_from_prompt(prompt)
                llm_calls += 1
                if ok and global_map[gx, gy] == 0:
                    waypoint = np.array([gx, gy])
                else:
                    waypoint = heuristic_fallback(top5, agent_pos, global_map)
                mode = "LLM"; wp_steps = 0
                pos_history.clear()

        # Act
        if mode == "LLM" and waypoint is not None:
            wp_steps += 1
            dist = np.linalg.norm(agent_pos - waypoint)
            if dist <= 1.5 or wp_steps > max_wp_steps:
                mode = "RL"; waypoint = None; pos_history.clear()
                action, _ = ppo_model.predict(obs, deterministic=True)
            else:
                wp_vec = (waypoint - agent_pos).astype(np.float32)
                wp_vec /= (np.linalg.norm(wp_vec) + 1e-8)
                fake_obs = np.concatenate([obs[:49], wp_vec, obs[51:]])
                action, _ = ppo_model.predict(fake_obs, deterministic=True)
        else:
            action, _ = ppo_model.predict(obs, deterministic=True)

        obs, r, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc:
            break

    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step, "llm_calls": llm_calls}


def run_episode_history(ppo_model, anomaly_type, seed):
    """Run one episode with History-Prompt LLM (trajectory coords)."""
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    pos_history   = []
    full_traj     = []
    anomaly_done  = False
    llm_calls     = 0
    mode          = "RL"
    waypoint      = None
    wp_steps      = 0
    patience      = 15
    stuck_thresh  = 1.5
    safe_dist     = 5.0
    max_wp_steps  = 25

    for step in range(2000):
        agent_pos  = env.agent_pos.copy()
        task_phase = env.task_phase
        anomaly    = info["anomaly_type"]
        global_map = info["global_map"]
        subtarget  = (info["item_pos"] if task_phase == 0
                      else info["goal_pos"])

        pos_history.append(agent_pos.copy())
        full_traj.append(tuple(agent_pos.astype(int)))
        if len(pos_history) > patience:
            pos_history.pop(0)

        def invoke_history_llm():
            nonlocal llm_calls, waypoint, mode, wp_steps
            grid_str, top5, top5_str, _ = build_ascii_grid(
                global_map, agent_pos, subtarget)
            dx = subtarget[0] - agent_pos[0]
            dy = subtarget[1] - agent_pos[1]
            prompt = build_history_prompt(
                grid_str, dx, dy, top5_str, full_traj, anomaly)
            gx, gy, ok = get_llm_waypoint_from_prompt(prompt)
            llm_calls += 1
            if ok and global_map[gx, gy] == 0:
                waypoint = np.array([gx, gy])
            else:
                waypoint = heuristic_fallback(top5, agent_pos, global_map)
            mode = "LLM"; wp_steps = 0

        if anomaly != ANOMALY_NONE and not anomaly_done:
            anomaly_done = True
            invoke_history_llm()
            pos_history.clear()

        if mode != "LLM" and (np.linalg.norm(agent_pos - subtarget) >= safe_dist
                               and len(pos_history) == patience):
            disp = np.linalg.norm(pos_history[-1] - pos_history[0])
            if disp < stuck_thresh:
                invoke_history_llm()
                pos_history.clear()

        if mode == "LLM" and waypoint is not None:
            wp_steps += 1
            dist = np.linalg.norm(agent_pos - waypoint)
            if dist <= 1.5 or wp_steps > max_wp_steps:
                mode = "RL"; waypoint = None; pos_history.clear()
                action, _ = ppo_model.predict(obs, deterministic=True)
            else:
                wp_vec = (waypoint - agent_pos).astype(np.float32)
                wp_vec /= (np.linalg.norm(wp_vec) + 1e-8)
                fake_obs = np.concatenate([obs[:49], wp_vec, obs[51:]])
                action, _ = ppo_model.predict(fake_obs, deterministic=True)
        else:
            action, _ = ppo_model.predict(obs, deterministic=True)

        obs, r, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc:
            break

    return {"success": env.task_phase == PHASE_DONE,
            "steps": env.current_step, "llm_calls": llm_calls}


# ── Main evaluation ────────────────────────────────────────────────────────
def evaluate():
    print("Loading PPO model...")
    ppo_model = PPO.load("models_saved/ppo_l3.zip", device="cpu")

    results = {"zero_shot": {}, "history_prompt": {}}

    for strategy, runner in [("zero_shot",     run_episode_zeroshot),
                              ("history_prompt", run_episode_history)]:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        for anomaly in ANOMALIES:
            aname = ANAMES[anomaly]
            print(f"\n  Anomaly: {aname}")
            episodes = []

            for ep in range(N_EPISODES):
                r = runner(ppo_model, anomaly, seed=ep * 100)
                episodes.append(r)
                print(f"    ep {ep+1:2d}/{N_EPISODES}"
                      f"  success={r['success']}"
                      f"  steps={r['steps']:4d}"
                      f"  llm={r['llm_calls']}")

            sr = np.mean([e['success'] for e in episodes]) * 100
            results[strategy][aname] = {
                "success_rate": sr,
                "avg_steps": float(np.mean(
                    [e['steps'] for e in episodes if e['success']]
                )) if any(e['success'] for e in episodes) else None,
                "avg_llm_calls": float(np.mean([e['llm_calls']
                                                 for e in episodes])),
                "n_episodes": N_EPISODES
            }
            print(f"  → SR={sr:.1f}%")

    # Load our existing results for comparison
    our_results = {}
    try:
        d = json.load(open(os.path.join(RESULTS_DIR,
                                         "exp_llm_gemma3_1b.json")))
        for aname in ["none","obstruction","displacement","invalidation"]:
            our_results[aname] = {
                "success_rate": d["pick_deliver_anomaly"][aname]["success_rate"]
            }
    except Exception:
        pass

    # Save
    out = {"zero_shot": results["zero_shot"],
           "history_prompt": results["history_prompt"],
           "ours_task_memory": our_results}
    path = os.path.join(RESULTS_DIR, "exp_llm_prompt_baselines.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    # Summary table
    anomaly_names = ["none","obstruction","displacement","invalidation"]
    print("\n" + "="*70)
    print("SUMMARY (L3, pick-and-deliver, 30 episodes)")
    print("="*70)
    print(f"{'Strategy':<22} {'None':>8} {'Obstruct':>10} {'Displace':>10} {'Invalid':>10} {'Avg':>8}")
    print("-"*70)
    for name, data in [("Zero-Shot LLM",   results["zero_shot"]),
                        ("History-Prompt",   results["history_prompt"]),
                        ("Ours (Task Mem.)", our_results)]:
        if not data:
            continue
        srs = [data.get(a, {}).get("success_rate", 0) for a in anomaly_names]
        avg = np.mean(srs)
        print(f"{name:<22} {srs[0]:>8.1f} {srs[1]:>10.1f}"
              f" {srs[2]:>10.1f} {srs[3]:>10.1f} {avg:>8.1f}")


if __name__ == "__main__":
    evaluate()
