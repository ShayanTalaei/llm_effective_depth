#!/bin/bash

python3 logitlens.py qwen3_32b
python3 logitlens.py qwen3_14b
python3 logitlens.py qwen3_8b

python3 analyze_future_local_effects.py qwen3_32b
python3 analyze_current_local_effects.py qwen3_32b

python3 analyze_future_local_effects.py qwen3_14b
python3 analyze_current_local_effects.py qwen3_14b

python3 analyze_future_local_effects.py qwen3_8b
python3 analyze_current_local_effects.py qwen3_8b

python3 skip_max_depth.py qwen3_32b math
python3 skip_max_depth.py qwen3_32b mquake
python3 analyze_future_effects.py qwen3_32b
python3 analyze_norms.py qwen3_32b 
python3 analyze_current_effects.py qwen3_32b
python3 igrad.py qwen3_32b
python3 analyze_token_skip.py qwen3_32b


python3 skip_max_depth.py qwen3_14b math
python3 skip_max_depth.py qwen3_14b mquake
python3 analyze_future_effects.py qwen3_14b
python3 analyze_norms.py qwen3_14b 
python3 analyze_current_effects.py qwen3_14b
python3 igrad.py qwen3_14b
python3 analyze_token_skip.py qwen3_14b


python3 skip_max_depth.py qwen3_8b math
python3 skip_max_depth.py qwen3_8b mquake
python3 analyze_future_effects.py qwen3_8b
python3 analyze_norms.py qwen3_8b 
python3 analyze_current_effects.py qwen3_8b
python3 igrad.py qwen3_8b
python3 analyze_token_skip.py qwen3_8b