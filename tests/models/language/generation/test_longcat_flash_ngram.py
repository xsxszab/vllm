# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial

from ....utils import multi_gpu_test
from ...registry import HF_EXAMPLE_MODELS
from ...utils import dummy_hf_overrides

ARCH = "LongcatFlashNgramForCausalLM"


def _two_layers(hf_config, **kwargs):
    hf_config = dummy_hf_overrides(hf_config, **kwargs)
    hf_config.update({"num_layers": 2})
    return hf_config


@multi_gpu_test(num_gpus=2)
def test_pipeline_parallel(vllm_runner, example_prompts) -> None:
    """Only the first PP stage holds the n-gram embedding: the second must take
    hidden states from the first instead of building n-gram embeddings."""
    info = HF_EXAMPLE_MODELS.get_hf_info(ARCH)
    info.check_available_online(on_fail="skip")
    info.check_transformers_version(on_fail="skip")

    with vllm_runner(
        info.default,
        load_format="dummy",
        hf_overrides=partial(
            _two_layers, model_arch=ARCH, exist_overrides=info.hf_overrides
        ),
        pipeline_parallel_size=2,
        enforce_eager=True,
        max_model_len=256,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(example_prompts, max_tokens=4)

    # Dummy weights make the tokens meaningless; both stages must run.
    assert len(outputs) == len(example_prompts)
