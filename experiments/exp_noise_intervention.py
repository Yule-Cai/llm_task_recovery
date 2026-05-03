# exp_noise_intervention.py
# Noise Intervention Experiment for KBS reviewer W2
# Tests system robustness when upstream anomaly signal contains errors
# Three noise types:
#   1. Wrong anomaly type (e.g., obstruction described as displacement)
#   2. Corrupted description (partial/garbled text)
#   3. Clean signal (baseline)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from models.arbiter import TaskArbiter
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT,
                            ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector

LLM_URL    = "http://172.20.0.1:1234/v1"
MAP_PATH   = "data/maps/L3_map0.csv"
N_EPISODES = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Noise injection definitions ───────────────────────────────────────────
# For each true anomaly, define wrong description alternatives
WRONG_DESCRIPTIONS = {
    ANOMALY_OBSTRUCTION: [
        # Describe obstruction as displacement
        "The target item has been moved to a new location. Navigate to find it.",
        # Describe obstruction as invalidation
        "The goal zone is now inaccessible. An alternate goal has been designated.",
    ],
    ANOMALY_DISPLACEMENT: [
        # Describe displacement as obstruction
        "New obstacles have appeared on your planned path. Find an alternative route.",
        # Describe displacement as invalidation
        "The goal zone is now inaccessible. An alternate goal has been designated.",
    ],
    ANOMALY_INVALIDATION: [
        # Describe invalidation as obstruction
        "New obstacles have appeared on your planned path. Find an alternative route.",
        # Describe invalidation as displacement
        "The target item has been moved to a new location. Navigate to find it.",
    ],
}

CORRUPTED_DESCRIPTIONS = {
    ANOMALY_OBSTRUCTION:  "Path... blocked... route... [SIGNAL DEGRADED]",
    ANOMALY_DISPLACEMENT: "Item... relocated... find... [SIGNAL DEGRADED]",
    ANOMALY_INVALIDATION: "Goal... unavailable... alternate... [SIGNAL DEGRADED]",
}

NOISE_CONDITIONS = {
    'clean':     {'noise_type': None,        'description': 'Clean signal (baseline)'},
    'wrong_50':  {'noise_type': 'wrong',     'noise_prob': 0.5,
                  'description': 'Wrong anomaly type (50% probability)'},
    'wrong_100': {'noise_type': 'wrong',     'noise_prob': 1.0,
                  'description': 'Wrong anomaly type (100% probability)'},
    'corrupt_50':{'noise_type': 'corrupted', 'noise_prob': 0.5,
                  'description': 'Corrupted description (50% probability)'},
    'corrupt_100':{'noise_type': 'corrupted','noise_prob': 1.0,
                  'description': 'Corrupted description (100% probability)'},
}

ANOMALIES = [ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANOMALY_LABELS = {
    ANOMALY_OBSTRUCTION:  'obstruction',
    ANOMALY_DISPLACEMENT: 'displacement',
    ANOMALY_INVALIDATION: 'invalidation',
}


def inject_signal_noise(true_anomaly, noise_type, noise_prob, rng):
    """Return (noisy_anomaly_label, noisy_description) for LLM prompt."""
    if noise_type is None:
        return true_anomaly, None  # None = use default description

    if rng.random() > noise_prob:
        return true_anomaly, None  # No noise this episode

    if noise_type == 'wrong':
        alternatives = WRONG_DESCRIPTIONS[true_anomaly]
        wrong_desc = rng.choice(alternatives)
        # Pick a wrong anomaly type label too
        wrong_anomaly = rng.choice([a for a in ANOMALIES if a != true_anomaly])
        return wrong_anomaly, wrong_desc

    elif noise_type == 'corrupted':
        return true_anomaly, CORRUPTED_DESCRIPTIONS[true_anomaly]

    return true_anomaly, None


def run_episode(ppo_model, true_anomaly, noise_anomaly, noise_desc, seed):
    """Run one episode with potentially noisy anomaly signal."""
    np.random.seed(seed)
    inj = AnomalyInjector(anomaly_type=true_anomaly, inject_at_step=100)
    env  = TaskEnv(map_path=MAP_PATH, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info = env._get_info()

    arbiter = TaskArbiter(
        rl_model=ppo_model,
        use_task_memory=True,
        llm_base_url=LLM_URL,
        override_anomaly_type=noise_anomaly,      # pass noisy signal
        override_anomaly_desc=noise_desc,          # pass noisy description
    )
    arbiter.reset(task_type=TASK_PICK_DELIVER)

    for _ in range(2000):
        action, mode = arbiter.predict(obs, info)
        obs, r, term, trunc, _ = flat.step(action)
        info = env._get_info()
        if term or trunc:
            break

    return {
        'success':   env.task_phase == PHASE_DONE,
        'steps':     env.current_step,
        'llm_calls': arbiter.llm_calls,
        'noise_applied': noise_desc is not None,
    }


def evaluate():
    print("Loading PPO model...")
    ppo_model = PPO.load("models_saved/ppo_l3.zip", device='cpu')
    rng = np.random.default_rng(42)

    results = {}

    for condition_name, condition in NOISE_CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"Condition: {condition['description']}")
        print(f"{'='*60}")
        results[condition_name] = {}

        for true_anomaly in ANOMALIES:
            aname = ANOMALY_LABELS[true_anomaly]
            print(f"\n  Anomaly: {aname}")
            episodes = []

            for ep in range(N_EPISODES):
                # Determine noise for this episode
                noise_type = condition.get('noise_type')
                noise_prob = condition.get('noise_prob', 0.0)
                noise_anomaly, noise_desc = inject_signal_noise(
                    true_anomaly, noise_type, noise_prob, rng)

                result = run_episode(
                    ppo_model, true_anomaly,
                    noise_anomaly, noise_desc, seed=ep*100)
                episodes.append(result)

                noisy = '⚡' if result['noise_applied'] else ' '
                print(f"    {noisy} ep {ep+1:2d}/{N_EPISODES}"
                      f"  success={result['success']}"
                      f"  steps={result['steps']:4d}"
                      f"  llm={result['llm_calls']}")

            sr           = np.mean([e['success'] for e in episodes]) * 100
            noise_count  = sum(1 for e in episodes if e['noise_applied'])
            noise_sr     = np.mean([e['success'] for e in episodes
                                    if e['noise_applied']]) * 100 \
                           if noise_count > 0 else None
            clean_sr     = np.mean([e['success'] for e in episodes
                                    if not e['noise_applied']]) * 100 \
                           if noise_count < N_EPISODES else None

            results[condition_name][aname] = {
                'success_rate':       sr,
                'noise_episodes':     noise_count,
                'noise_sr':           noise_sr,
                'clean_sr_in_mixed':  clean_sr,
                'avg_llm_calls':      np.mean([e['llm_calls'] for e in episodes]),
            }
            print(f"  → SR={sr:.1f}%  noise_episodes={noise_count}"
                  + (f"  noise_SR={noise_sr:.1f}%" if noise_sr else "")
                  + (f"  clean_SR={clean_sr:.1f}%" if clean_sr else ""))

    # Save results
    out = os.path.join(RESULTS_DIR, 'exp_noise_intervention.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Condition':<20} {'Obstr.':<10} {'Displ.':<10} {'Inval.':<10} {'Avg':<10}")
    print("-"*70)
    for cname, cdata in results.items():
        srs = [cdata[a]['success_rate'] for a in ['obstruction','displacement','invalidation']]
        avg = np.mean(srs)
        print(f"{cname:<20} {srs[0]:<10.1f} {srs[1]:<10.1f} {srs[2]:<10.1f} {avg:<10.1f}")


if __name__ == '__main__':
    evaluate()