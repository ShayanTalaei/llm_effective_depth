# Analysis for our paper "Do Language Models Use Their Depth Efficiently?"

Please set environment variable `NDIF_TOKEN` to your NDIF token. If you don't have one and you want to run only the local models, set it to empty string.

See `run_qwen.sh` on how to run the experiments. The output PDF plots will be generated in the `out` subdirectory.

## Analyzing the finetuned DeepMind Math models

Please use either `python3 dm_math_igrad.py <checkpoint name>`, `python3 dm_math_igrad.py <checkpoint name>` or `python3 future_effects_dm_math.py <checkpoint name>`` with the stripped checkpoint (just the "model" dict element extracted according to the description in the training code).

## Fitting layers vs. performance

Run `open_llm_leaderboard_stat.py` and `plot_llm_capabilities.py`. The fit parameters for the Open LLM Leaderboard will be printed on the terminal.