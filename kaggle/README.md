# Foundation Labs: running nanochat on Kaggle

This fork runs nanochat on Kaggle's free GPUs (T4) instead of a rented 8xH100 machine.

- `pilot/` : smallest end-to-end run (6-layer model). Proves every stage works and times each one.
  Push: `kaggle kernels push -p kaggle/pilot --accelerator NvidiaTeslaT4 -t 10800`
  Output: `kaggle kernels output lifamaziya/nanochat-pilot -p <folder>` (pilot_report.json, pilot_log.txt, checkpoints)

T4 notes: no bf16, so `NANOCHAT_DTYPE=float16`. Kaggle sessions stop at ~12 h, so longer runs use `--save-every` and `--resume-from-step`.
