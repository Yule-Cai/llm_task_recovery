# run_all.py
# Runs exp_search_only.py then exp_react_baseline.py sequentially
# with unified progress tracking
# Run from project root: python run_all.py

import subprocess, sys, time, os

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"  Starting: {script_name}")
    print(f"  Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    proc = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()

    status = "OK" if proc.returncode == 0 else f"ERROR (code {proc.returncode})"
    print(f"\n{'='*60}")
    print(f"  Finished: {script_name}  [{status}]")
    print(f"  Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    return proc.returncode

if __name__ == "__main__":
    t0 = time.time()
    print(f"Starting pipeline at {time.strftime('%H:%M:%S')}")
    print("Step 1/2: Search task rerun (no LLM needed, ~30 min)")
    rc1 = run_script("experiments/exp_search_only.py")

    if rc1 != 0:
        print("exp_search_only.py failed — aborting pipeline.")
        sys.exit(1)

    print("Step 2/2: CoT prompt baseline (~2 hrs, needs LM Studio)")
    rc2 = run_script("experiments/exp_react_baseline.py")

    elapsed = time.time() - t0
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\nPipeline complete in {h}h {m}m {s}s")
    sys.exit(rc2)
