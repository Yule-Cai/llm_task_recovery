# train_lstm.py
# Trains a RecurrentPPO (LSTM-PPO) baseline for comparison with Task Memory RL-LLM
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import sys
import time
import torch
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from envs.task_env import TaskEnv

STEPS_PER_LEVEL  = 3_000_000
SAVE_DIR         = os.path.join(project_root, "models_saved")
CHECKPOINT_DIR   = os.path.join(project_root, "models_saved", "checkpoints_lstm")
MAP_DIR          = os.path.join(project_root, "data", "maps")
TRAIN_LEVELS     = ["L1", "L2", "L3"]
NUM_ENVS         = 8


class ProgressCallback(BaseCallback):
    def __init__(self, total_steps, level, bar_width=40):
        super().__init__()
        self.total_steps = total_steps
        self.level = level
        self.bar_width = bar_width
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print("")

    def _on_rollout_end(self):
        steps = self.num_timesteps
        elapsed = time.time() - self.start_time
        fps = steps / elapsed if elapsed > 0 else 0
        remain = (self.total_steps - steps) / fps if fps > 0 else 0
        pct = steps / self.total_steps
        filled = int(self.bar_width * pct)
        bar = "#" * filled + "-" * (self.bar_width - filled)

        def fmt(s):
            h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
            return (str(h)+"h " if h > 0 else "") + str(m)+"m "+str(sec)+"s"

        print("\r[LSTM-"+self.level+"] ["+bar+"] "+str(round(pct*100,1))+"%  "+
              str(steps)+"/"+str(self.total_steps)+
              "  fps="+str(int(fps))+"  ETA="+fmt(remain),
              end="", flush=True)
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        h = int(elapsed//3600); m = int((elapsed%3600)//60); sec = int(elapsed%60)
        print("\r[LSTM-"+self.level+"] ["+"#"*self.bar_width+"] 100.0%  "+
              str(self.total_steps)+"/"+str(self.total_steps)+
              "  Total: "+str(h)+"h "+str(m)+"m "+str(sec)+"s")

    def _on_step(self):
        return True


def make_env(map_path, rank):
    def _init():
        env = TaskEnv(map_path=map_path, max_steps=2000,
                      anomaly_injector=None, task_type=None)
        return gym.wrappers.FlattenObservation(env)
    return _init


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    grand_start = time.time()
    current_model = None

    for idx, level in enumerate(TRAIN_LEVELS):
        print("\n" + "="*60)
        print("  LSTM-PPO Level "+str(idx+1)+"/"+str(len(TRAIN_LEVELS))+": "+level)
        print("="*60)

        map_path = os.path.join(MAP_DIR, level+"_map0.csv")
        if not os.path.exists(map_path):
            print("Map not found:", map_path); continue

        vec_env = SubprocVecEnv([make_env(map_path, i) for i in range(NUM_ENVS)])

        checkpoint_cb = CheckpointCallback(
            save_freq=max(500_000//NUM_ENVS, 1),
            save_path=os.path.join(CHECKPOINT_DIR, level),
            name_prefix="lstm_ppo_"+level, verbose=0)
        progress_cb = ProgressCallback(STEPS_PER_LEVEL, level)

        if current_model is None:
            print("Initialising LSTM-PPO from scratch...")
            model = RecurrentPPO(
                "MlpLstmPolicy", vec_env,
                verbose=0,
                n_steps=2048,
                batch_size=256,
                learning_rate=3e-4,
                device="cpu",
                policy_kwargs=dict(
                    lstm_hidden_size=256,
                    n_lstm_layers=1,
                    shared_lstm=False,
                )
            )
        else:
            print("Loading from:", current_model)
            model = RecurrentPPO.load(current_model, env=vec_env,
                                      device="cpu",
                                      custom_objects={"verbose": 0})

        model.learn(total_timesteps=STEPS_PER_LEVEL,
                    callback=[checkpoint_cb, progress_cb],
                    reset_num_timesteps=False)

        save_path = os.path.join(SAVE_DIR, "lstm_ppo_"+level.lower())
        model.save(save_path)
        current_model = save_path
        print("Saved:", save_path+".zip")
        vec_env.close()

    elapsed = time.time() - grand_start
    h = int(elapsed//3600); m = int((elapsed%3600)//60); s = int(elapsed%60)
    print("\nLSTM training complete. Total: "+str(h)+"h "+str(m)+"m "+str(s)+"s")


if __name__ == "__main__":
    train()
