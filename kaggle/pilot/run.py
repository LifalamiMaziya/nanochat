# Foundation Labs: nanochat pilot on Kaggle (1x T4).
# Goal: prove the full pipeline runs (tokenizer -> pre-train -> eval -> SFT -> chat)
# and measure how long each stage takes, so bigger runs can be planned.
import os, subprocess, time, json, shutil

REPO = "https://github.com/LifalamiMaziya/nanochat.git"
W = "/kaggle/working"
SRC = "/tmp/nanochat"
BASE = f"{W}/nanochat_cache"          # checkpoints + tokenizer land in the output
LOG = f"{W}/pilot_log.txt"
report = {"stages": {}}

env = dict(os.environ,
           NANOCHAT_BASE_DIR=BASE,
           NANOCHAT_DTYPE="float16",   # T4 has no bf16
           OMP_NUM_THREADS="1",
           CUDA_VISIBLE_DEVICES="0",   # single GPU for the pilot
           PYTHONPATH=SRC,              # run from the repo, no package install
           WANDB_MODE="disabled")

def stage(name, cmd, cwd=SRC):
    print(f"\n===== {name} =====", flush=True)
    t = time.time()
    with open(LOG, "a") as log:
        log.write(f"\n===== {name} =====\n$ {cmd}\n"); log.flush()
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    secs = round(time.time() - t, 1)
    report["stages"][name] = {"seconds": secs, "ok": p.returncode == 0}
    json.dump(report, open(f"{W}/pilot_report.json", "w"), indent=2)
    print(f"{name}: {secs}s rc={p.returncode}", flush=True)
    if p.returncode != 0:
        os.system(f"tail -60 {LOG}")
        raise SystemExit(f"stage failed: {name}")

subprocess.run("nvidia-smi", shell=True)
stage("clone", f"git clone -q --depth 1 {REPO} {SRC}", cwd="/tmp")
stage("install", "pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 && pip install -q filelock kernels numpy psutil pyarrow rustbpe tiktoken wandb")
stage("data", "python -m nanochat.dataset -n 4")
stage("tokenizer", "python -m scripts.tok_train --max-chars=1000000000 && python -m scripts.tok_eval")
stage("pretrain_d6", "python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L "
      "--max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 "
      "--eval-every=200 --eval-tokens=262144 --core-metric-every=-1 --sample-every=500 "
      "--num-iterations=1000 --save-every=500 --run=dummy")
stage("base_eval", "python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16")
stage("sft", "python -m scripts.chat_sft --eval-every=200 --eval-tokens=262144 --num-iterations=300 --chatcore-every=-1 --run=dummy")
stage("chat_sample", "python -m scripts.chat_cli -p 'Hello! What can you do?' && "
      "python -m scripts.chat_cli -p 'What is the capital of France?'")

# keep the output small: drop raw data shards, keep tokenizer + checkpoints + logs
shutil.rmtree(f"{BASE}/base_data", ignore_errors=True)
for d in os.listdir(BASE):
    p = os.path.join(BASE, d)
    if os.path.isdir(p) and "data" in d: shutil.rmtree(p, ignore_errors=True)
print(json.dumps(report, indent=2))
