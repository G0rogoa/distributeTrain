import unittest

from train_perf.nccl import parse_nccl_output

SAMPLE = """# nccl-tests synthetic sanitized fixture following v2.17.5 text columns
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
  1024 256 float sum -1 8.50 0.12 0.18 0 7.90 0.13 0.19 0
  2048 512 float sum -1 9.00 0.23 0.34 0 8.10 0.25 0.37 0
"""


class NcclParserTest(unittest.TestCase):
    def test_separates_in_place(self):
        rows = parse_nccl_output(SAMPLE)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["in_place"] for row in rows}, {True, False})
        self.assertEqual(rows[0]["bytes"], 1024)

    def test_empty_is_error(self):
        with self.assertRaisesRegex(ValueError, "no complete"):
            parse_nccl_output("# truncated")

    def test_truncated_row_is_error(self):
        with self.assertRaisesRegex(ValueError, "no complete"):
            parse_nccl_output("1024 256 float sum -1 8.50 0.12")

    def test_correctness_error_is_error(self):
        with self.assertRaisesRegex(ValueError, "non-zero"):
            parse_nccl_output(SAMPLE.replace("0 7.90", "1 7.90", 1))


if __name__ == "__main__":
    unittest.main()
