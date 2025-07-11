"""
Model Loading Utilities for Language Model Analysis

This module provides a unified interface for loading language models from different sources:
1. Remote models via NDIF (Neural Data Infrastructure) API
2. Local models loaded directly with Transformers/nnsight

The system automatically handles:
- Model quantization for memory efficiency
- Device mapping for multi-GPU setups  
- Different model architectures (Llama, Qwen, etc.)
- Fallback from remote to local when needed

This abstraction allows analysis scripts to work with any supported model
without worrying about the loading mechanics.
"""

from nnsight import LanguageModel
import torch

# Remote models: accessed via NDIF API for efficient inference
# These models run on remote servers, saving local GPU memory
remote_model_table = {
    # "llama_3.1_8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama_3.1_70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama_3.1_405b": "meta-llama/Meta-Llama-3.1-405B",
}

# Local models: loaded directly on local hardware
# Format: "model_name": ("huggingface_path", use_quantization)
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
    """Get the Hugging Face model path for a remote model."""
    return remote_model_table[model_name]

def create_model(model_name, force_local=False):
    """
    Create a language model for analysis, choosing the best loading strategy.
    
    Loading priority:
    1. If force_local=False and model is in remote_model_table: Use NDIF remote access
    2. If model is in local_model_table: Load locally with optional quantization
    3. Otherwise: Raise error
    
    Args:
        model_name: String identifier for the model (e.g., "qwen3_32b")
        force_local: If True, never use remote access even if available
        
    Returns:
        LanguageModel: nnsight model wrapper ready for analysis
        
    The returned model has a .remote attribute indicating whether it uses remote inference.
    """
    if (not force_local) and (model_name in remote_model_table):
        # Use NDIF remote access - model runs on remote servers
        # Advantages: No local GPU memory usage, access to very large models
        # Disadvantages: Network latency, requires NDIF token
        model = LanguageModel(remote_model_table[model_name], device_map="auto")
        model.remote = True
        
    elif model_name in local_model_table:
        # Load model locally on available hardware
        model_path, use_quantization = local_model_table[model_name]
        
        if use_quantization:
            # Use 8-bit quantization to reduce memory usage
            # This allows loading larger models on smaller GPUs with minimal accuracy loss
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
            )
        else:
            bnb_config = None

        # Load with nnsight wrapper for analysis capabilities
        model = LanguageModel(
            model_path, 
            device_map="cuda:0",              # Automatically distribute across available GPUs
            torch_dtype=torch.bfloat16,     # Use bfloat16 for memory efficiency
            quantization_config=bnb_config, # Apply quantization if enabled
            dispatch=False                  # Don't use Accelerate dispatching (nnsight handles this)
        )
        
        model.remote = False
    else:
        raise ValueError(f"Model {model_name} not found in remote_model_table or local_model_table")

    return model
