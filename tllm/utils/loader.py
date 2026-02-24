import os, json, torch
from safetensors.torch import load_file

def load_state_dict(path: str) -> dict[str, torch.Tensor]:


