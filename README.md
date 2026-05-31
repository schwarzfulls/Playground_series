# 目的

IDごとに次の周回でピットインするかを予測する．

## 提出フォーマット

ROC 曲線の下面積

## カラムの意味

- id
  - ユニーク
- Driver
  - 特定の名称ではなく，英数字で示される
- Compound
- Race
  - レースの名称
- Year
  - 年
- PitStop
  - タイヤ交換、燃料補給、修理、セッティング調整などのためにpitに行くこと
- LapNumber
  - マシンによるレースの周回数
- Stint
  - レース終了までのピットストップの回数
- TyreLife
  - タイヤライフ(タイヤを交換してから，タイヤの性能が落ちて交換するまでに必要な距離[km?])
- Position
- LapTime (s)
  - マシンが1周走るのにかかった時間
- LapTime_Delta
  - 基準となるラップタイムと現在のラップとの時間差
- Cumulative_Degradation
  - タイヤの劣化具合の数値？
- RaceProgress
  - レースの進行具合？
- Position_Change
  - 順位の入れ替わり
    - マイナスの意味
- PitNextLap
  - 次の周に pit in する意思表示
    - 1と0(する/しない)
