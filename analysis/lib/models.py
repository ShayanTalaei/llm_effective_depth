from nnsight import LanguageModel
import torch

remote_model_table = {
    "llama_3.1_8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama_3.1_70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama_3.1_405b": "meta-llama/Meta-Llama-3.1-405B",
}

local_model_table = {
    "qwen2.5_14b": ("Qwen/Qwen2.5-14B", True),
    "qwen2.5_7b": ("Qwen/Qwen2.5-7B", True),
    "qwen2.5_1.5b": ("Qwen/Qwen2.5-1.5B", True),
    "qwen3_8b": ("Qwen/Qwen3-8B", True),
    "qwen3_14b": ("Qwen/Qwen3-14B", True),
    "qwen3_32b": ("Qwen/Qwen3-32B", True),
    "llama_3.1_8b": ("meta-llama/Meta-Llama-3.1-8B", True),
    "llama_3.1_405b": ("meta-llama/Meta-Llama-3.1-405B", True),
    "llama_3.1_8b_instruct": ("meta-llama/Meta-Llama-3.1-8B-Instruct", True),
    "llama_3.1_70b_instruct": ("meta-llama/Meta-Llama-3.1-70B-Instruct", True),
}

def get_model(model_name):
    return remote_model_table[model_name]

def create_model(model_name, force_local=False):
    if (not force_local) and (model_name in remote_model_table):
        model = LanguageModel(remote_model_table[model_name], device_map="auto")
        model.remote = True
    elif model_name in local_model_table:
        if local_model_table[model_name][1]:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        model = LanguageModel(
            local_model_table[model_name][0], device_map="auto", torch_dtype=torch.bfloat16, 
            quantization_config=bnb_config, dispatch=False)
        
        model.remote = False
    else:
        raise ValueError(f"Model {model_name} not found")

    return model
