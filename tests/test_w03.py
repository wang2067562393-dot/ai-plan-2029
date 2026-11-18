# -*- coding: utf-8 -*-
"""w03 纯函数测试：用合成小数据，不依赖网络与 titanic.csv。"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from w03_survival_analysis import build_markdown, survival_by  # noqa: E402


def toy_df() -> pd.DataFrame:
    # 舱位1：2 人生还 1 人 -> 0.5；舱位2：2 人全灭 -> 0.0
    return pd.DataFrame(
        {
            "Pclass": [1, 1, 2, 2],
            "Sex": ["male", "female", "male", "female"],
            "Survived": [0, 1, 0, 0],
        }
    )


class TestSurvival(unittest.TestCase):
    def test_group_rates_and_sort(self):
        t = survival_by(toy_df(), ["Pclass"])
        self.assertEqual(list(t["Pclass"]), [1, 2])  # 按生还率降序
        self.assertAlmostEqual(t.loc[0, "survival_rate"], 0.5)
        self.assertEqual(list(t["count"]), [2, 2])

    def test_multi_index_group(self):
        t = survival_by(toy_df(), ["Pclass", "Sex"])
        self.assertEqual(len(t), 4)
        top = t.iloc[0]
        self.assertEqual((top["Pclass"], top["Sex"]), (1, "female"))
        self.assertAlmostEqual(top["survival_rate"], 1.0)

    def test_markdown(self):
        md = build_markdown(survival_by(toy_df(), ["Pclass"]))
        self.assertIn("Pclass", md.splitlines()[0])
        self.assertIn("0.500", md)


if __name__ == "__main__":
    unittest.main()
