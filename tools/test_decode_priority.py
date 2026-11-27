#!/usr/bin/env python3
"""Unit tests for tools/gen_decode_priority.py (no GPU needed)."""
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import gen_decode_priority as gdp


def git_stock(relpath: str) -> str:
    return subprocess.check_output(
        ["git", "-C", REPO, "show", f"HEAD:{relpath}"], text=True
    )


class TestDecodePriorityGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        revs = subprocess.check_output(
            ["git", "-C", REPO, "rev-list", "--max-count=20", "HEAD"], text=True
        ).splitlines()
        for rev in revs:
            p = subprocess.check_output(
                ["git", "-C", REPO, "show", f"{rev}:package/program/prepare.loom"], text=True
            )
            if gdp.DP_PREP_TABLE_BEGIN not in p:
                cls.stock_prep = p
                cls.stock_ctrl = subprocess.check_output(
                    ["git", "-C", REPO, "show", f"{rev}:package/program/control.loom"], text=True
                )
                return
        raise RuntimeError("could not locate stock prepare.loom in recent git history")

    def test_patch_defaults_and_markers(self):
        p = gdp.patch_prepare(self.stock_prep)
        c = gdp.patch_control(self.stock_ctrl)
        self.assertIn(gdp.DP_PREP_HELPER_BEGIN, p)
        self.assertIn(gdp.DP_PREP_TABLE_BEGIN, p)
        self.assertIn("view<9x5xi32>", p)
        self.assertIn("index.constant 1840 : offset", p)
        self.assertIn(gdp.DP_CTRL_ENC_BEGIN, c)
        self.assertIn(gdp.DP_CTRL_PUB_BEGIN, c)
        self.assertIn("index.constant 2020 : offset", c)
        self.assertIn("cadence=5, budget=512", c)
        # Ensure we stay strictly below the 2048-byte sampling record header
        self.assertLess(1840 + 9 * 5 * 4, 2048)
        self.assertLessEqual(2020 + 4 * 4, 2048)

    def test_double_patch_rejected(self):
        p = gdp.patch_prepare(self.stock_prep)
        c = gdp.patch_control(self.stock_ctrl)
        with self.assertRaises(ValueError):
            gdp.patch_prepare(p)
        with self.assertRaises(ValueError):
            gdp.patch_control(c)

    def test_param_validation(self):
        with self.assertRaises(ValueError):
            gdp.patch_control(self.stock_ctrl, cadence=0)
        with self.assertRaises(ValueError):
            gdp.patch_control(self.stock_ctrl, budget=1024)
        with self.assertRaises(ValueError):
            gdp.patch_control(self.stock_ctrl, small_max=200, big_thresh=128)

    def test_budget_256_and_cadence_1(self):
        c = gdp.patch_control(self.stock_ctrl, cadence=1, budget=256)
        self.assertIn("cadence=1, budget=256", c)
        self.assertIn("%dp_reset_ctr = scalar.constant 0 : i32", c)

    def test_repo_programs_match_generator(self):
        with open(os.path.join(REPO, "package/program/prepare.loom")) as f:
            cur_p = f.read()
        with open(os.path.join(REPO, "package/program/control.loom")) as f:
            cur_c = f.read()
        self.assertEqual(cur_p, gdp.patch_prepare(self.stock_prep))
        self.assertEqual(cur_c, gdp.patch_control(self.stock_ctrl, cadence=5, budget=512))


if __name__ == "__main__":
    unittest.main()
