import unittest

from sglang.srt.layers.attention.dsa.utils import (
    dsa_in_seq_cp_extend_lens_are_segment_aligned,
)


class TestDSACPInSequenceSegmentAlignment(unittest.TestCase):
    def test_frozen_cp8_shapes_are_admitted(self):
        for lengths in ([90_000], [19_616], [10_016], [80_384]):
            with self.subTest(lengths=lengths):
                self.assertTrue(
                    dsa_in_seq_cp_extend_lens_are_segment_aligned(lengths, 8)
                )

    def test_padding_does_not_make_real_rows_cp_safe(self):
        for lengths, cp_size in (
            ([9_428], 8),
            ([0], 8),
            ([10_016, 9_428], 8),
            ([10_016], 0),
        ):
            with self.subTest(lengths=lengths, cp_size=cp_size):
                self.assertFalse(
                    dsa_in_seq_cp_extend_lens_are_segment_aligned(lengths, cp_size)
                )


if __name__ == "__main__":
    unittest.main()
