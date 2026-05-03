# exp_nlp_model_comparison.py
# Tests BERT/RoBERTa/GPT-2 class models as cognitive layer replacements
# These models are not designed for coordinate generation - results document why generative LLMs are necessary
# Run: python exp_nlp_model_comparison.py (no LM Studio needed)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json, re, numpy as np, time

# HuggingFace mirror for faster downloads in China
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import gymnasium as gym

from stable_baselines3 import PPO
from envs.task_env import (TaskEnv, TASK_PICK_DELIVER, PHASE_DONE,
                            ANOMALY_NONE, ANOMALY_OBSTRUCTION,
                            ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION)
from envs.anomaly_injector import AnomalyInjector
from models.task_memory import TaskMemory

MAP_PATHS   = {
    "L1": "data/maps/L1_map0.csv",
    "L2": "data/maps/L2_map0.csv",
    "L3": "data/maps/L3_map0.csv",
}
N_EPISODES  = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ANOMALIES = [ANOMALY_NONE, ANOMALY_OBSTRUCTION,
             ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION]
ANAMES    = {ANOMALY_NONE:"none", ANOMALY_OBSTRUCTION:"obstruction",
             ANOMALY_DISPLACEMENT:"displacement",
             ANOMALY_INVALIDATION:"invalidation"}

# Models to test - mix of encoder and decoder architectures
MODELS = [
    # ── English-only encoders (cannot generate coordinates) ──────────────
    #{"name": "distilbert-base-uncased",  "type": "encoder", "hf_id": "distilbert-base-uncased"},
    #{"name": "tinybert",                 "type": "encoder", "hf_id": "huawei-noah/TinyBERT_General_4L_312D"},
    #{"name": "mobilebert",               "type": "encoder", "hf_id": "google/mobilebert-uncased"},
    #{"name": "albert-base-v2",           "type": "encoder", "hf_id": "albert-base-v2"},
    # ── Multilingual encoders ─────────────────────────────────────────────
    #{"name": "mbert",                    "type": "encoder", "hf_id": "bert-base-multilingual-cased"},
    #{"name": "xlm-roberta-base",         "type": "encoder", "hf_id": "xlm-roberta-base"},
    # ── Small generative LLMs ─────────────────────────────────────────────
    #{"name": "gpt2-medium",              "type": "decoder", "hf_id": "gpt2-medium"},
    #{"name": "TinyLlama-1.1B",           "type": "decoder", "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
   # {"name": "SmolLM2-1.7B",             "type": "decoder", "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},
    #{"name": "Qwen2.5-0.5B",             "type": "decoder", "hf_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    {"name": "Qwen2.5-1.5B",             "type": "decoder", "hf_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    {"name": "Phi-3.5-mini",             "type": "decoder", "hf_id": "microsoft/Phi-3.5-mini-instruct"},
]


def load_model(model_info):
    """Load model and tokenizer from HuggingFace."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    import torch

    hf_id = model_info["hf_id"]
    mtype = model_info["type"]
    print(f"  Loading {hf_id}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if mtype == "decoder":
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
        else:
            # Encoder-only: use masked LM head for text generation attempt
            from transformers import AutoModelForMaskedLM
            try:
                model = AutoModelForMaskedLM.from_pretrained(hf_id, trust_remote_code=True)
            except:
                model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        return tokenizer, model
    except Exception as e:
        print(f"  Failed to load {hf_id}: {e}")
        return None, None


def get_waypoint(tokenizer, model, model_info, prompt, global_map, agent_pos):
    """Attempt to get waypoint from model. Returns waypoint or None."""
    import torch

    mtype = model_info["type"]

    try:
        if mtype == "decoder":
            # Generative models: try to generate coordinate
            inputs = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            # Decode only new tokens
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

        else:
            # Encoder-only: cannot generate, record failure
            reply = "[ENCODER MODEL - CANNOT GENERATE COORDINATES]"

        # Try to extract [X, Y] from reply
        m = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', reply)
        if m:
            gx, gy = int(m.group(1)), int(m.group(2))
            if (0 <= gx < global_map.shape[0] and
                0 <= gy < global_map.shape[1] and
                global_map[gx, gy] == 0):
                return np.array([gx, gy]), reply[:80], True

        return None, reply[:80], False

    except Exception as e:
        return None, f"ERROR: {e}", False


def build_prompt(global_map, agent_pos, subtarget, task_memory, anomaly):
    """Build waypoint generation prompt."""
    H, W = global_map.shape
    ax, ay = int(agent_pos[0]), int(agent_pos[1])
    r = 3  # same as PPO and LLM in main framework (7x7 window)
    min_x, max_x = max(0, ax-r), min(H, ax+r+1)
    min_y, max_y = max(0, ay-r), min(W, ay+r+1)
    local = global_map[min_x:max_x, min_y:max_y]

    grid_str = ""
    candidates = []
    dx_g = subtarget[0]-ax; dy_g = subtarget[1]-ay
    for i in range(local.shape[0]):
        for j in range(local.shape[1]):
            if i == ax-min_x and j == ay-min_y:
                grid_str += "R "
            elif local[i,j] == 1:
                grid_str += "X "
            else:
                gi, gj = i+min_x, j+min_y
                dot = (gi-ax)*dx_g + (gj-ay)*dy_g
                if dot >= 0:
                    grid_str += ". "
                    candidates.append((gi, gj))
                else:
                    grid_str += "- "
        grid_str += "\n"

    candidates.sort(key=lambda p: np.linalg.norm(np.array(p)-np.array(subtarget)))
    top5 = candidates[:5]
    top5_str = ", ".join(f"[{p[0]},{p[1]}]" for p in top5)

    lines = [
        "Robot navigation task. Output the best waypoint coordinate.",
        f"Grid (R=Robot, .=free, X=wall): {grid_str.strip()}",
        f"Target direction: dx={dx_g:.0f}, dy={dy_g:.0f}",
    ]
    if task_memory:
        lines.append(f"Task: {task_memory.to_prompt_string(anomaly_type=anomaly)}")
    lines.append(f"Safe candidates: {top5_str}")
    lines.append("Output ONLY [X, Y]:")

    return "\n".join(lines), top5


def run_episode(ppo_model, tokenizer, hf_model, model_info,
                anomaly_type, seed, map_path="data/maps/L3_map0.csv"):
    np.random.seed(seed)
    inj = (AnomalyInjector(anomaly_type=anomaly_type, inject_at_step=100)
           if anomaly_type != ANOMALY_NONE else None)
    env  = TaskEnv(map_path=map_path, max_steps=2000,
                   anomaly_injector=inj, task_type=TASK_PICK_DELIVER)
    flat = gym.wrappers.FlattenObservation(env)
    obs, _ = flat.reset()
    info   = env._get_info()

    task_memory = TaskMemory(task_type=TASK_PICK_DELIVER)
    pos_history = []; anomaly_done = False
    llm_calls = 0; mode = "RL"; waypoint = None
    wp_steps = 0; patience = 15; stuck_thresh = 1.5
    safe_dist = 5.0; max_wp_steps = 25
    model_replies = []

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
            prompt, top5 = build_prompt(global_map, agent_pos,
                                         subtarget, task_memory, anomaly)
            wp, reply, success = get_waypoint(tokenizer, hf_model,
                                               model_info, prompt,
                                               global_map, agent_pos)
            model_replies.append(reply)
            llm_calls += 1
            if wp is not None and success:
                waypoint = wp
                mode = "LLM"; wp_steps = 0
            else:
                # Model cannot generate coords - disable stuck detector
                # to prevent infinite triggering, pure PPO takes over
                waypoint = None
                mode = "RL"  # PPO takes over, no waypoint injection

        if anomaly != ANOMALY_NONE and not anomaly_done:
            anomaly_done = True; invoke(); pos_history.clear()

        # Stuck detector disabled for NLP model comparison
        # Only anomaly trigger is used - this isolates the model's ability
        # to handle semantic anomalies, not spatial navigation

        if mode == "LLM" and waypoint is not None:
            wp_steps += 1
            if np.linalg.norm(agent_pos-waypoint)<=1.5 or wp_steps>max_wp_steps:
                mode="RL"; waypoint=None; pos_history.clear()
                action, _ = ppo_model.predict(obs, deterministic=True)
            else:
                wp_vec = (waypoint-agent_pos).astype(np.float32)
                wp_vec /= (np.linalg.norm(wp_vec)+1e-8)
                mod_obs = np.concatenate([obs[:49], wp_vec, obs[51:]])
                action, _ = ppo_model.predict(mod_obs, deterministic=True)
        else:
            action, _ = ppo_model.predict(obs, deterministic=True)

        obs, _, term, trunc, _ = flat.step(int(action))
        info = env._get_info()
        if term or trunc: break

    return {
        "success": env.task_phase == PHASE_DONE,
        "steps": env.current_step,
        "llm_calls": llm_calls,
        "sample_reply": model_replies[0] if model_replies else "",
        "coord_success_rate": sum(1 for r in model_replies
                                   if re.search(r'\[\d+,\s*\d+\]', r)) / max(len(model_replies), 1)
    }


def evaluate():
    print("Loading PPO model...")
    ppo = PPO.load("models_saved/ppo_l3.zip", device="cpu")

    all_results = {}
    total_models = len(MODELS)

    for mi, model_info in enumerate(MODELS):
        mname = model_info["name"]
        print(f"\n{'='*60}")
        print(f"  Model {mi+1}/{total_models}: {mname} ({model_info['type']})")
        print(f"{'='*60}")

        tokenizer, hf_model = load_model(model_info)
        if tokenizer is None:
            print(f"  Skipping {mname} - failed to load")
            all_results[mname] = {"error": "load_failed"}
            continue

        model_results = {}
        total_eps = len(ANOMALIES) * N_EPISODES
        done_eps  = 0

        for level, map_path in MAP_PATHS.items():
          for anomaly in ANOMALIES:
            aname = ANAMES[anomaly]
            episodes = []

            for ep in range(N_EPISODES):
                r = run_episode(ppo, tokenizer, hf_model,
                                model_info, anomaly, seed=ep*100,
                                map_path=map_path)
                episodes.append(r)
                done_eps += 1
                pct = done_eps / total_eps * 100
                bar = "#"*int(pct/2) + "-"*(50-int(pct/2))
                print(f"  [{bar}] {pct:5.1f}%  {aname}  ep{ep+1}  "
                      f"success={r['success']}  coord_hit={r['coord_success_rate']:.0%}",
                      flush=True)

            sr = np.mean([e['success'] for e in episodes]) * 100
            coord_rate = np.mean([e['coord_success_rate'] for e in episodes])
            key = f"{level}_{aname}"
            model_results[key] = {
                "success_rate": sr,
                "avg_llm_calls": float(np.mean([e['llm_calls'] for e in episodes])),
                "coord_generation_rate": float(coord_rate),
                "sample_reply": episodes[0]['sample_reply']
            }
            print(f"  → SR={sr:.1f}%  coord_rate={coord_rate:.0%}")

        # Compute avg SR across all levels and anomalies
        all_srs = [v['success_rate'] for k,v in model_results.items()
                   if isinstance(v, dict) and 'success_rate' in v]
        avg_sr = np.mean(all_srs) if all_srs else 0.0
        model_results['avg_sr'] = avg_sr
        model_results['model_type'] = model_info['type']
        all_results[mname] = model_results

        # Save intermediate results
        path = os.path.join(RESULTS_DIR, 'exp_nlp_model_comparison.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f'  Saved intermediate results to {path}')

        # Free memory
        del hf_model, tokenizer
        import gc; gc.collect()

    # Save
    path = os.path.join(RESULTS_DIR, "exp_nlp_model_comparison.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {path}")

    # Summary table
    print("\n" + "="*75)
    print(f"{'Model':<25} {'Type':<10} {'None':>7} {'Obstr':>7} "
          f"{'Displ':>7} {'Inval':>7} {'Avg':>7} {'CoordHit':>10}")
    print("-"*75)
    for mname, res in all_results.items():
        if 'error' in res: continue
        all_srs = [v['success_rate'] for k,v in res.items()
                   if isinstance(v,dict) and 'success_rate' in v]
        all_coord = [v.get('coord_generation_rate',0) for k,v in res.items()
                     if isinstance(v,dict) and 'success_rate' in v]
        avg = np.mean(all_srs) if all_srs else 0
        coord = np.mean(all_coord) if all_coord else 0
        print(f"{mname:<25} {res.get('model_type','?'):<10} "
              f"avg={avg:>6.1f}%  coord_hit={coord:>6.0%}")


if __name__ == "__main__":
    evaluate()