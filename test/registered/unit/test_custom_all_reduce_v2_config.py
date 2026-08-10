import unittest

from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    resolve_max_pull_size,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCustomAllReduceV2Config(unittest.TestCase):
    def test_default_preserves_16_mib_limit(self):
        envs.SGLANG_JIT_CUSTOM_ALL_REDUCE_MAX_PULL_SIZE_BYTES.clear()
        self.assertEqual(resolve_max_pull_size(None), 16 * 1024 * 1024)

    def test_environment_can_raise_limit(self):
        with envs.SGLANG_JIT_CUSTOM_ALL_REDUCE_MAX_PULL_SIZE_BYTES.override(
            128 * 1024 * 1024
        ):
            self.assertEqual(resolve_max_pull_size(None), 128 * 1024 * 1024)

    def test_explicit_constructor_value_wins(self):
        with envs.SGLANG_JIT_CUSTOM_ALL_REDUCE_MAX_PULL_SIZE_BYTES.override(123):
            self.assertEqual(resolve_max_pull_size(456), 456)

    def test_negative_environment_value_is_rejected(self):
        with envs.SGLANG_JIT_CUSTOM_ALL_REDUCE_MAX_PULL_SIZE_BYTES.override(-1):
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                resolve_max_pull_size(None)


if __name__ == "__main__":
    unittest.main()
