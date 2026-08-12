"""CANN ACLNN random-number baseline adapters."""

from __future__ import annotations

from typing import Any

import torch


class CannAclnnGenerator:
    """Persistent torch_npu generator routed to CANN ACLNN in-place RNG ops."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator(device="npu")
        self.generator.manual_seed(seed)

    def generate(self, distribution: str, output: torch.Tensor) -> torch.Tensor:
        if distribution == "raw32":
            output.random_(
                -(1 << 31),
                1 << 31,
                generator=self.generator,
            )
        elif distribution.startswith("uniform"):
            output.uniform_(0.0, 1.0, generator=self.generator)
        elif distribution.startswith("normal"):
            output.normal_(0.0, 1.0, generator=self.generator)
        elif distribution.startswith("lognormal"):
            # CANN 9.0 exposes no in-place LogNormal ACLNN symbol.  The strict
            # baseline is therefore Normal plus Exp, both inside the timed call.
            output.normal_(0.0, 1.0, generator=self.generator)
            output.exp_()
        else:
            raise ValueError(f"Unsupported CANN baseline distribution: {distribution}")
        return output


def baseline_metadata(distribution: str) -> dict[str, Any]:
    if distribution == "raw32":
        components = ["torch.Tensor.random_", "aclnnInplaceRandom"]
    elif distribution.startswith("uniform"):
        components = ["torch.Tensor.uniform_", "aclnnInplaceUniform"]
    elif distribution.startswith("normal"):
        components = ["torch.Tensor.normal_", "aclnnInplaceNormal"]
    else:
        components = [
            "torch.Tensor.normal_",
            "aclnnInplaceNormal",
            "torch.Tensor.exp_",
            "CANN elementwise Exp",
        ]
    return {
        "implementation_path": "torch_npu_op_plugin_to_cann_aclnn",
        "components": components,
        "comparison_scope": "statistical_and_performance_not_bitstream_equivalent",
    }


__all__ = ["CannAclnnGenerator", "baseline_metadata"]
