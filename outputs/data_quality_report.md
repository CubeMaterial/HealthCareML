# Data Quality Report

## 연도별 행/컬럼 수
| year | rows | columns | missing_columns |
| --- | --- | --- | --- |
| 2020 | 229269 | 18 | height, weight |
| 2021 | 229242 | 18 |  |
| 2022 | 231785 | 18 |  |
| 2023 | 231752 | 18 | sleep_weekday, sleep_weekend, sleep_avg |
| 2024 | 231728 | 18 |  |
| 2025 | 231615 | 18 |  |

## 사용된 공통 변수
year, age, sex, height, weight, bmi, smoking, drinking, physical_activity, walking, sleep_weekday, sleep_weekend, sleep_avg, stress, subjective_health, diabetes_label, hypertension_label, obesity_label

## Label 클래스 분포
### obesity_label
| obesity_label | count |
| --- | --- |
| 0.0 | 945207.0 |
| 1.0 | 420632.0 |
| nan | 19552.0 |

### diabetes_label
| diabetes_label | count |
| --- | --- |
| 0.0 | 1200268.0 |
| 1.0 | 185008.0 |
| nan | 115.0 |

### hypertension_label
| hypertension_label | count |
| --- | --- |
| 0.0 | 961474.0 |
| 1.0 | 423791.0 |
| nan | 126.0 |

## 결측률 상위 컬럼
| column | missing_rate |
| --- | --- |
| walking | 0.33747079344387254 |
| smoking | 0.3055101411803599 |
| height | 0.17404833725641353 |
| weight | 0.17046956418801623 |
| sleep_weekend | 0.16772593441129616 |
| sleep_weekday | 0.1676501435334862 |
| sleep_avg | 0.16757146538414064 |
| physical_activity | 0.023123435910872816 |
| bmi | 0.014112983266095997 |
| obesity_label | 0.014112983266095997 |

## BMI 생성 방식
- 2020은 `oba_bmi`를 사용합니다.
- 2021~2025는 `height`, `weight`가 있으면 직접 계산합니다.
- 2023~2024처럼 `oba_bmi`가 함께 있으면 직접 계산값을 우선하고, 없을 때 `oba_bmi`를 보조로 사용합니다.
- `bmi < 10` 또는 `bmi > 60`은 결측 처리합니다.

## 수면 변수 누락 연도
- 2023년은 `mtc_17z1`, `mtc_18z1`가 없어 `sleep_avg`가 NaN입니다.

## Leakage 방지 규칙
- 비만 모델 feature에서 `bmi`, `height`, `weight`를 제외합니다.
- 당뇨 모델 feature에서 `dia_*` 계열을 제외합니다.
- 고혈압 모델 feature에서 `hya_*` 계열을 제외합니다.
- 진단/치료 문항은 label 생성에만 사용합니다.