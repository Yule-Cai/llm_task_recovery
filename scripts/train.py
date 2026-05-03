# scripts/train.py
import os
import sys
import time
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from envs.task_env import TaskEnv

STEPS_PER_LEVEL  = 10_000_000
SAVE_DIR         = os.path.join(project_root, "models_saved")
CHECKPOINT_DIR   = os.path.join(project_root, "models_saved", "checkpoints")
MAP_DIR          = os.path.join(project_root, "data", "maps")
TRAIN_LEVELS     = ["L1", "L2", "L3", "L4", "L5"]
START_FROM_MODEL = None
NUM_ENVS         = 16


class ProgressCallback(BaseCallback):
    def __init__(self, total_steps, level, bar_width=40):
        super().__init__()
        self.total_steps = total_steps
        self.level       = level
        self.bar_width   = bar_width
        self.start_time  = None

    def _on_training_start(self):
        self.start_time = time.time()
        print("")

    def _on_rollout_end(self):
        steps   = self.num_timesteps
        elapsed = time.time() - self.start_time
        fps     = steps / elapsed if elapsed > 0 else 0
        remain  = (self.total_steps - steps) / fps if fps > 0 else 0

        pct    = steps / self.total_steps
        filled = int(self.bar_width * pct)
        bar    = "#" * filled + "-" * (self.bar_width - filled)

        def fmt(s):
            h   = int(s // 3600)
            m   = int((s % 3600) // 60)
            sec = int(s % 60)
            if h > 0:
                return str(h) + "h " + str(m) + "m " + str(sec) + "s"
            return str(m) + "m " + str(sec) + "s"

        print(
            "\r[" + self.level + "] [" + bar + "] " +
            str(round(pct * 100, 1)) + "%  " +
            str(steps) + "/" + str(self.total_steps) +
            "  fps=" + str(int(fps)) +
            "  ETA=" + fmt(remain),
            end="", flush=True
        )
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        h   = int(elapsed // 3600)
        m   = int((elapsed % 3600) // 60)
        sec = int(elapsed % 60)
        bar = "#" * self.bar_width
        print(
            "\r[" + self.level + "] [" + bar + "] 100.0%  " +
            str(self.total_steps) + "/" + str(self.total_steps) +
            "  Total: " + str(h) + "h " + str(m) + "m " + str(sec) + "s"
        )

    def _on_step(self):
        return True


def make_env(map_path, rank):
    def _init():
        env = TaskEnv(
            map_path=map_path,
            max_steps=2000,
            anomaly_injector=None,
            task_type=None,
        )
        return gym.wrappers.FlattenObservation(env)
    return _init


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    current_model_path = START_FROM_MODEL
    total_levels       = len(TRAIN_LEVELS)
    grand_start        = time.time()

    for idx, level in enumerate(TRAIN_LEVELS):
        print("\n" + "=" * 60)
        print("  Level " + str(idx + 1) + "/" + str(total_levels) +
              ": " + level + "  (" + str(STEPS_PER_LEVEL) + " steps)")
        print("=" * 60)

        map_path = os.path.join(MAP_DIR, level + "_map0.csv")
        if not os.path.exists(map_path):
            print("Map not found: " + map_path + " - skipping")
            continue

        vec_env = SubprocVecEnv([make_env(map_path, i) for i in range(NUM_ENVS)])

        checkpoint_cb = CheckpointCallback(
            save_freq   = max(500_000 // NUM_ENVS, 1),
            save_path   = os.path.join(CHECKPOINT_DIR, level),
            name_prefix = "ppo_" + level,
            verbose     = 0,
        )
        progress_cb = ProgressCallback(
            total_steps = STEPS_PER_LEVEL,
            level       = level,
        )

        model_zip = (
            current_model_path + ".zip"
            if current_model_path and not current_model_path.endswith(".zip")
            else current_model_path
        )

        if model_zip is None or not os.path.exists(model_zip):
            print("Initialising new policy from scratch...")
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose       = 0,
                n_steps       = 4096,
                batch_size    = 1024,
                learning_rate = 3e-4,
                device        = "cpu",
            )
        else:
            print("Loading weights from: " + model_zip)
            model = PPO.load(
                model_zip, env=vec_env, device="cpu",
                custom_objects={"verbose": 0},
            )

        model.learn(
            total_timesteps     = STEPS_PER_LEVEL,
            callback            = [checkpoint_cb, progress_cb],
            reset_num_timesteps = False,
        )

        save_path = os.path.join(SAVE_DIR, "ppo_" + level.lower())
        model.save(save_path)
        current_model_path = save_path
        print("Saved: " + save_path + ".zip")
        vec_env.close()

    grand_elapsed = time.time() - grand_start
    h   = int(grand_elapsed // 3600)
    m   = int((grand_elapsed % 3600) // 60)
    s   = int(grand_elapsed % 60)
    print("\nAll levels complete. Total: " +
          str(h) + "h " + str(m) + "m " + str(s) + "s")


if __name__ == "__main__":
    train()