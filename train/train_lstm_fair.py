# train_lstm_fair.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, multiprocessing, time

MAX_CORES = multiprocessing.cpu_count()
print(f"[System] CPU cores: {MAX_CORES}")

# Intel XPU
try:
    import intel_extension_for_pytorch as ipex
    import torch
    torch.set_num_threads(MAX_CORES)
    if torch.xpu.is_available():
        USE_XPU = True
        print(f"[XPU] ✓ Intel Arc GPU active: {torch.xpu.get_device_name(0)}")
    else:
        USE_XPU = False
        print("[XPU] ✗ Not available, using CPU")
except Exception as e:
    USE_XPU = False
    import torch
    torch.set_num_threads(MAX_CORES)
    print(f"[XPU] ✗ Failed ({e}), using CPU")

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from envs.task_env import TaskEnv, TASK_PICK_DELIVER

LEVELS       = ['L2', 'L3']
MAP_TEMPLATE = "data/maps/{}_map0.csv"
SAVE_DIR     = "models_saved"
STEPS_LEVEL  = 10_000_000
L1_N_ENVS    = 16        # L1 model was trained with 16 envs
N_ENVS       = 16        # optimal for SubprocVecEnv on this CPU
os.makedirs(SAVE_DIR, exist_ok=True)


def make_env(map_path, task_type=TASK_PICK_DELIVER, max_steps=2000):
    def _init():
        import gymnasium as gym
        env = TaskEnv(map_path=map_path, max_steps=max_steps,
                      task_type=task_type)
        return gym.wrappers.FlattenObservation(env)
    return _init


class LiveProgress(BaseCallback):
    def __init__(self, total, level):
        super().__init__(0)
        self.total = total
        self.level = level
        self.t0 = time.time()          # start immediately
        self.last_print_time = 0
        self.last_time = self.t0
        self.last_steps = 0
        self.ema_fps = 0
        self.start_steps = None        # set on first step

    def _on_training_start(self):
        # reset timer when training actually starts
        self.t0 = time.time()
        self.last_time = self.t0
        self.start_steps = self.model.num_timesteps

    def format_time(self, seconds):
        if seconds < 0: return "00:00"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    def _on_step(self):
        now = time.time()
        if now - self.last_print_time >= 0.5:
            done = self.model.num_timesteps - self.start_steps
            if done <= 0: return True
            elapsed = now - self.t0
            delta_steps = done - self.last_steps
            delta_time  = now - self.last_time
            current_fps = delta_steps / delta_time if delta_time > 0 else 0
            self.ema_fps = (0.7 * self.ema_fps + 0.3 * current_fps
                            if self.ema_fps > 0 else current_fps)
            self.last_time = now
            self.last_steps = done
            eta_sec = (self.total - done) / max(self.ema_fps, 1)
            pct = min(max(done / self.total * 100, 0), 100)
            bar = '█' * int(pct/2) + '░' * (50 - int(pct/2))
            dev = 'XPU' if USE_XPU else 'CPU'
            print(f'\r[{bar}] {pct:5.1f}%  fps={int(self.ema_fps):<5}  {dev}  '
                  f'已用:{self.format_time(elapsed)}  '
                  f'剩余:{self.format_time(eta_sec)}  L={self.level}',
                  end='', flush=True)
            self.last_print_time = now
        return True

    def _on_training_end(self):
        print()


def train():
    # Load L1 model first (with original 16 envs)
    model = None
    l1_path = os.path.join(SAVE_DIR, 'lstm_fair_L1.zip')
    if os.path.exists(l1_path):
        print(f'Loading L1 model: {l1_path}')
        # Load with a temp env matching L1's 16 envs
        tmp_env = SubprocVecEnv([make_env("data/maps/L1_map0.csv")
                                  for _ in range(L1_N_ENVS)])
        model = RecurrentPPO.load(l1_path, env=tmp_env,
                    device='xpu' if USE_XPU else 'cpu')
        tmp_env.close()

    for level in LEVELS:
        map_path = MAP_TEMPLATE.format(level)
        print(f"\n{'='*60}")
        print(f"  Level: {level}  Envs: {N_ENVS}  Device: {'XPU' if USE_XPU else 'CPU'}")
        print(f"{'='*60}")

        env = SubprocVecEnv([make_env(map_path) for _ in range(N_ENVS)])

        if model is None:
            model = RecurrentPPO(
                "MlpLstmPolicy", env,
                n_steps=512, batch_size=512,
                learning_rate=3e-4, gamma=0.99,
                n_epochs=10, verbose=0,
                device='xpu' if USE_XPU else 'cpu',
            )
        else:
            # Re-load with new env size to allow different N_ENVS
            save_tmp = os.path.join(SAVE_DIR, '_tmp_model.zip')
            model.save(save_tmp)
            model = RecurrentPPO.load(save_tmp, env=env,
                          device='xpu' if USE_XPU else 'cpu')

        model.verbose = 0
        save_cb = CheckpointCallback(
            save_freq=max(STEPS_LEVEL // (10 * N_ENVS), 1),
            save_path=SAVE_DIR,
            name_prefix=f"lstm_fair_{level}",
            verbose=0
        )

        model.learn(
            total_timesteps=STEPS_LEVEL,
            callback=[LiveProgress(STEPS_LEVEL, level), save_cb],
            reset_num_timesteps=False,
            progress_bar=False,
        )

        save_path = os.path.join(SAVE_DIR, f"lstm_fair_{level}.zip")
        model.save(save_path)
        print(f"✓ Saved: {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    train()