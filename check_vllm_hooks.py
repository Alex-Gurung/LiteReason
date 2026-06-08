#!/usr/bin/env python3
"""Check that the LiteReason vLLM plugin's patch points exist in the installed vLLM.

`vllm_plugin.py` monkeypatches private vLLM v1 internals by name, so a vLLM upgrade
can move them. Run this after bumping vLLM:

    python check_vllm_hooks.py

A moved class fails its import with a precise error; a changed method fails an
assert naming the method and the params it lost. Instance attributes the patches
read at runtime (e.g. input_batch.num_tokens_no_spec, query_start_loc.np,
inputs_embeds.gpu, execute_model_state) are exercised by run_smoke.sh.
"""
import inspect

import vllm
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.activation import GeluAndMul, SiluAndMul
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# Imported above only to confirm they still live where the plugin expects them;
# referenced here so the imports aren't flagged as unused.
_PLUGIN_DEPS = (
    set_current_vllm_config, GeluAndMul, SiluAndMul, ColumnParallelLinear, RowParallelLinear,
)


def require_params(cls, method, *params):
    """Assert `cls.method` exists and still accepts the params the plugin passes."""
    sig = inspect.signature(getattr(cls, method))
    missing = [p for p in params if p not in sig.parameters]
    assert not missing, f"{cls.__name__}.{method} changed: missing {missing} (has {list(sig.parameters)})"


require_params(GPUModelRunner, "_preprocess", "scheduler_output", "num_input_tokens", "intermediate_tensors")
require_params(GPUModelRunner, "execute_model")
require_params(GPUModelRunner, "_update_states_after_model_execute")
require_params(DefaultModelLoader, "load_weights", "model", "model_config")
require_params(Gemma3ForConditionalGeneration, "load_weights", "weights")

print(f"OK: vllm {vllm.__version__} — plugin hook points present.")
