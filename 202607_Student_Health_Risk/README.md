# 目的

## 提出フォーマット

## カラムの意味

- id: Int64
- health_condition: String
- sleep_duration: Float64
  - null_count: 75999 / 690088 = 11%
- heart_rate: Float64
  - null_count: 7833 / 690088 = 1%
- bmi: Float64
  - null_count: 13898 / 690088 = 2%
- calorie_expenditure: Float64
  - null_count: 52853 / 690088 = 8%
- step_count: Float64
  - null_count: 13916 / 690088 = 2%
- exercise_duration: Float64
  - null_count: 6901 / 690088 = 1%
- water_intake: Float64
  - null_count: 43477 / 690088 = 6%
- diet_type: String
  - null_count: 6901 / 690088 = 1%
- stress_level: String
  - null_count: 82811 / 690088 = 12%
- sleep_quality: String
  - null_count: 58331 / 690088 = 8%
- physical_activity_level: String
  - null_count: 36621 / 690088 = 5%
- smoking_alcohol: String
  - null_count: 28582 / 690088 = 4%
- gender: String
  - null_count: 21373 / 690088 = 3%

## 用語の整理

- sleep_duration
  - 睡眠時間[h]
- heart_rate
  - 心拍数(下)
- bmi
  - 身長[m] / 体重[kg]\*2
- calorie_expenditure
  - カロリー消費量[kcal]
- step_count
  - 歩数[歩]
- exercise_duration
  - 運動時間[h?min?]
- water_intake
  - 水分摂取量[L]
- diet_type
  - veg: 野菜
  - balanced: バランスよく
  - non-veg: 非野菜
- stress_level
  - high
  - low
  - medium
- sleep_quality
  - average
  - poor
  - good
- physical_activity_level
  - moderate: 適度
  - sedentary: 座りがち
  - active: よく動く
- smoking_alcohol
  - yes
  - no
  - occasional
- gender
  - male
  - female
  - other

## その他の情報

null の取り扱いに注意する -> 特徴量重要度が高いカラムにおいて null がある場合は，その id の予測精度は低くなりそう．
