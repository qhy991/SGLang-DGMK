"""CPU contract tests for the B300 masked SwiGLU quant campaign switch."""

from __future__ import annotations

import os
import unittest

from sglang.srt.layers.glm52_opt import config


_ENV_NAMES = (
    "SGLANG_GLM52_OPT",
    "SGLANG_GLM52_OPT_PROFILE",
    "SGLANG_GLM52_OPT_OPS",
    "SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT",
)


def _set_campaign_env(
    *,
    enabled: bool = True,
    profile: str = "combined_winners",
    ops: str | None = "moe_swiglu_quant",
    variant: str | None = "cuda_valid_cta",
) -> None:
    os.environ["SGLANG_GLM52_OPT"] = "1" if enabled else "0"
    os.environ["SGLANG_GLM52_OPT_PROFILE"] = profile
    for name, value in (
        ("SGLANG_GLM52_OPT_OPS", ops),
        ("SGLANG_OPT_MOE_SWIGLU_QUANT_VARIANT", variant),
    ):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    config.swiglu_quant_variant.cache_clear()


class SwigluQuantConfigTest(unittest.TestCase):
    def setUp(self):
        self.saved = {name: os.environ.get(name) for name in _ENV_NAMES}
        self.env_applied = config._env_applied
        # These tests own os.environ directly and do not exercise the optional
        # worker side-channel file loader.
        config._env_applied = True

    def tearDown(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        config._env_applied = self.env_applied
        config.swiglu_quant_variant.cache_clear()

    def test_candidate_requires_every_explicit_gate(self):
        _set_campaign_env()
        self.assertEqual(config.swiglu_quant_variant(), "cuda_valid_cta")

        _set_campaign_env(enabled=False)
        self.assertIsNone(config.swiglu_quant_variant())

        _set_campaign_env(profile="full")
        self.assertIsNone(config.swiglu_quant_variant())

        _set_campaign_env(ops=None)
        self.assertIsNone(config.swiglu_quant_variant())

        _set_campaign_env(ops="moe_gate_proj,moe_down_proj")
        self.assertIsNone(config.swiglu_quant_variant())

        _set_campaign_env(variant=None)
        self.assertIsNone(config.swiglu_quant_variant())

    def test_candidate_is_explicit_only_in_combined_winners(self):
        _set_campaign_env(ops=None)
        self.assertNotIn("moe_swiglu_quant", config.combined_winner_ops())

        _set_campaign_env(ops="moe_swiglu_quant")
        self.assertEqual(config.combined_winner_ops(), {"moe_swiglu_quant"})
        self.assertEqual(config.swiglu_quant_variant(), "cuda_valid_cta")

    def test_candidate_rejects_unknown_variant(self):
        _set_campaign_env(variant="typo")
        with self.assertRaisesRegex(ValueError, "cuda_valid_cta"):
            config.swiglu_quant_variant()


if __name__ == "__main__":
    unittest.main()
