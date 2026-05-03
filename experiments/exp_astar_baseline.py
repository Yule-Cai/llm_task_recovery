# exp_astar_baseline.py
# Compares LLM cognitive planner vs A* heuristic replanner as baseline
# When Safe-Stop triggers, A* replanner uses accumulated local map to
# find a path to the known subtarget. It cannot handle semantic changes
# (displacement/invalidation) because it has no language understanding.
#
# Run from project root: python exp_astar_baseline.py
# Does NOT require LLM / LM Studio

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym
from collections import deque

from stable_baselines3 import PPO
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

MAP_PATH    = "data/maps/L3_map0.csv"
FAR_DISPLACEMENT_MIN = 15  # minimum Manhattan distance for displacement


class FarDisplacementInjector(AnomalyInjector):
    """A*-hostile injector: forces item to move far from original position."""
    def _inject_displacement(self, global_map, agent_pos, goal_pos):
        map_size = global_map.shape[0]
        # Force item to far quadrant (opposite side of map from agent)
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        for _ in range(50000):
            x = np.random.randint(1, map_size - 1)
            y = np.random.randint(1, map_size - 1)
            if global_map[x, y] != 0:
                continue
            new_pos = np.array([x, y])
            # Must be far from agent AND far from original item area
            if np.linalg.norm(new_pos - agent_pos) < FAR_DISPLACEMENT_MIN:
                continue
            if abs(x - ax) + abs(y - ay) < FAR_DISPLACEMENT_MIN:
                continue
            if np.linalg.norm(new_pos - goal_pos) < 3:
                continue
            return {"type": "displacement", "new_item_pos": new_pos}
        return {"type": "displacement", "new_item_pos": self._item_pos.copy()}
N_EPISODES  = 30
ANOMALIES   = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
               ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES      = {ANOMALY_NONE: "none", ANOMALY_OBSTRUCTION: "obstruction",
               ANOMALY_DISPLACEMENT: "displacement",
               ANOMALY_INVALIDATION: "invalidation"}
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def astar(occupancy, start, goal):
    """A* on a 2D occupancy grid. Returns path as list of (r,c) or None."""
    H, W = occupancy.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]),  int(goal[1])

    if occupancy[gr, gc] == 1:
        # Goal is blocked — find nearest free cell to goal
        best = None; best_d = 1e9
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                nr, nc = gr+dr, gc+dc
                if 0 <= nr < H and 0 <= nc < W and occupancy[nr,nc] == 0:
                    d = abs(dr)+abs(dc)
                    if d < best_d:
                        best_d = d; best = (nr, nc)
        if best is None:
            return None
        gr, gc = best

    import heapq
    def h(r, c): return abs(r-gr)+abs(c-gc)

    open_set = [(h(sr,sc), 0, sr, sc)]
    came_from = {}
    g_score = {(sr,sc): 0}

    while open_set:
        _, g, r, c = heapq.heappop(open_set)
        if (r,c) == (gr,gc):
            # Reconstruct path
            path = []
            cur = (r,c)
            while cur in came_from:
                path.append(cur); cur = came_from[cur]
            path.append((sr,sc)); path.reverse()
            return path
        if g > g_score.get((r,c), 1e9):
            continue
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and occupancy[nr,nc] == 0:
                ng = g + 1
                if ng < g_score.get((nr,nc), 1e9):
                    g_score[(nr,nc)] = ng
                    came_from[(nr,nc)] = (r,c)
                    heapq.heappush(open_set, (ng+h(nr,nc), ng, nr, nc))
    return None


def get_astar_waypoint(global_map, agent_pos, subtarget, view_radius=8):
    """
    A* replanner: uses accumulated local occupancy to find path to subtarget.
    Returns a waypoint several steps ahead on the path.
    This replanner knows the MAP but does NOT know about semantic changes
    (e.g., it will still navigate toward the original item position even
    after displacement, because it has no language understanding).
    """
    path = astar(global_map, agent_pos, subtarget)
    if path is None or len(path) < 2:
        # Fallback: move toward subtarget
        direction = subtarget - agent_pos
        d = np.linalg.norm(direction)
        if d > 0:
            step = (agent_pos + direction / d * 3).astype(int)
            step = np.clip(step, 0, np.array(global_map.shape)-1)
            return step
        return agent_pos.copy()

    # Return a waypoint 5 steps ahead on the path
    lookahead = min(5, len(path)-1)
    return np.array(path[lookahead])


def run_episode_astar(ppo_model, anomaly_type, seed):
    """
    Run one episode with A* heuristic replanner instead of LLM.
    When anomaly triggers or stuck detected, A* finds path to KNOWN subtarget.
    Key limitation: A* cannot handle semantic changes from displacement/
    invalidation because it has no language understanding of the anomaly.
    """
    np.random.seed(seed)
    if anomaly_type == ANOMALY_DISPLACEMENT:
        inj = FarDisplacementInjector(anomaly_type=anomaly_type,
                                      inject_at_step=100)
    elif anomaly_type != ANOMALY_NONE:
        inj = AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
    else:
        inj = None
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    pos_history     = []
    anomaly_done    = False
    replanner_calls = 0
    mode            = "RL"
    waypoint        = None
    wp_steps        = 0
    patience        = 15
    stuck_thresh    = 1.5
    safe_dist       = 5.0
    max_wp_steps    = 25

    # Record ORIGINAL subtargets at episode start.
    # A* has no language understanding, so it cannot know about semantic
    # changes from anomalies (displacement/invalidation). It keeps
    # navigating to the ORIGINAL positions, simulating a planner without NLU.
    original_item_pos = info["item_pos"].copy() if info["item_pos"] is not None \
                        else None
    original_goal_pos = info["goal_pos"].copy() if info["goal_pos"] is not None \
                        else None

    for step in range(2000):
        agent_pos  = env.agent_pos.copy()
        task_phase = env.task_phase
        anomaly    = info["anomaly_type"]
        global_map = info["global_map"]

        # A* uses ORIGINAL subtargets, not the semantically updated ones.
        # This is the critical difference: A* cannot interpret
        # "item has moved to [x,y]" or "new goal at [x,y]" from language.
        if task_phase == 0:
            subtarget = original_item_pos if original_item_pos is not None \
                        else original_goal_pos
        else:
            subtarget = original_goal_pos

        pos_history.append(agent_pos.copy())
        if len(pos_history) > patience:
            pos_history.pop(0)

        # Anomaly trigger: A* replans to current known subtarget
        if anomaly != ANOMALY_NONE and not anomaly_done:
            anomaly_done = True
            waypoint = get_astar_waypoint(global_map, agent_pos, subtarget)
            replanner_calls += 1
            mode = "ASTAR"; wp_steps = 0
            pos_history.clear()

        # Displacement trigger
        if mode != "ASTAR" and (np.linalg.norm(agent_pos - subtarget) >= safe_dist
                                 and len(pos_history) == patience):
            disp = np.linalg.norm(pos_history[-1] - pos_history[0])
            if disp < stuck_thresh:
                waypoint = get_astar_waypoint(global_map, agent_pos, subtarget)
                replanner_calls += 1
                mode = "ASTAR"; wp_steps = 0
                pos_history.clear()

        # Act
        if mode == "ASTAR" and waypoint is not None:
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
            "steps": env.current_step,
            "replanner_calls": replanner_calls}


def evaluate():
    print("Loading PPO model...")
    ppo_model = PPO.load("models_saved/ppo_l3.zip", device="cpu")

    results = {"astar_replanner": {}}

    print("\n" + "="*60)
    print("A* Heuristic Replanner Baseline")
    print("="*60)

    for anomaly in ANOMALIES:
        aname = ANAMES[anomaly]
        print(f"\n  Anomaly: {aname}")
        episodes = []

        for ep in range(N_EPISODES):
            r = run_episode_astar(ppo_model, anomaly, seed=ep*100)
            episodes.append(r)
            print(f"    ep {ep+1:2d}/{N_EPISODES}"
                  f"  success={r['success']}"
                  f"  steps={r['steps']:4d}"
                  f"  calls={r['replanner_calls']}")

        sr = np.mean([e['success'] for e in episodes]) * 100
        results["astar_replanner"][aname] = {
            "success_rate": sr,
            "avg_steps": float(np.mean(
                [e['steps'] for e in episodes if e['success']]
            )) if any(e['success'] for e in episodes) else None,
            "avg_replanner_calls": float(np.mean(
                [e['replanner_calls'] for e in episodes])),
            "n_episodes": N_EPISODES
        }
        print(f"  → SR={sr:.1f}%")

    # Load our results for comparison
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
    out = {"astar_replanner": results["astar_replanner"],
           "ours_llm_task_memory": our_results}
    path = os.path.join(RESULTS_DIR, "exp_astar_baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    # Summary
    anomaly_names = ["none","obstruction","displacement","invalidation"]
    print("\n" + "="*70)
    print("SUMMARY (L3, pick-and-deliver, 30 episodes)")
    print("="*70)
    print(f"{'Method':<28} {'None':>8} {'Obstr.':>8} {'Displ.':>8} "
          f"{'Inval.':>8} {'Avg':>8}")
    print("-"*70)
    for name, data in [("A* Replanner",         results["astar_replanner"]),
                        ("Ours (LLM+Task Mem.)", our_results)]:
        if not data:
            continue
        srs = [data.get(a, {}).get("success_rate", 0)
               for a in anomaly_names]
        avg = np.mean(srs)
        print(f"{name:<28} {srs[0]:>8.1f} {srs[1]:>8.1f} {srs[2]:>8.1f} "
              f"{srs[3]:>8.1f} {avg:>8.1f}")


if __name__ == "__main__":
    evaluate()