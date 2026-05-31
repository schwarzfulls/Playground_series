r"""catboost と LightGBM のアンサンブル(相加平均)して，csv ファイルを作成する

結果
スコアは: 0.93975

結論
LightGBM は，catboost の弱点を補完できていない
-> catboost 単体の方が精度が高い

次は，catboost のハイパラチューニングを行う

"""

# /// script
# dependencies = [
#     "polars",
# ]
# ///

from pathlib import Path

import polars as pl

OUTPUT = Path("./output/ensemble/")

cat = pl.read_csv("./output/GBDT/catboost/submission_catboost_first.csv")
lgb = pl.read_csv("./output/GBDT/lightbgm/submission_lightgbm_first.csv")

df = cat.join(
    lgb,
    on="id",
    suffix="_lgb",
)

df = df.with_columns(((pl.col("PitNextLap") + pl.col("PitNextLap_lgb")) / 2).alias("PitNextLap"))

OUTPUT.mkdir(exist_ok=True)

df.select(["id", "PitNextLap"]).write_csv("./output/ensemble/submission_ensemble.csv")
df.select(["id", "PitNextLap"]).write_csv(OUTPUT / "submission_ensemble.csv")
