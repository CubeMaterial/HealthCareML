# HealthCareNote Risk Dataset Pipeline

질병관리청 지역사회건강조사 2020~2025 원시자료를 앱 설문 기반 ML 모델 학습용 데이터셋으로 정리하는 프로젝트입니다.

현재 목표는 포트폴리오용 건강 위험도 예측 앱 프로토타입에 사용할 `비만`, `당뇨`, `고혈압` 학습 데이터셋을 만드는 것입니다. 의료 진단 목적이 아니며, 원자료 이용지침서/코드북 기준의 설문 응답 정리와 label leakage 방지가 핵심입니다.

## 구조

```text
config/
  variable_mapping.yaml        # 공통 변수명, 연도별 원본 컬럼명, 변수별 전처리 규칙
scripts/
  01 inspect_columns.py        # 원자료 컬럼/행 수 점검
  02 build_common_dataset.py   # 연도별 clean CSV와 통합 clean CSV 생성
  03 make_model_datasets.py    # 비만/당뇨/고혈압 모델별 데이터셋 생성
  04 train_baseline_models.py  # RandomForest/XGBoost/LightGBM 학습 및 성능 비교 리포트 생성
  05 inspect_model_sizes.py    # 모델 파일 크기 확인
  06 export_onnx_models.py     # 질환별 최고 모델을 ONNX로 변환하고 앱 assets에 복사
  common.py                    # 공통 입출력/전처리 유틸
outputs/
  columns_2020.csv             # inspect 결과
  data_quality_report.md       # 품질 리포트
  model_reports/               # 모델 평가 리포트
Data/
  processed/                   # 연도별 clean CSV
  ml/                          # 통합 데이터와 모델별 학습 데이터셋
models/                        # joblib 모델 파일
```

## 실행 순서

```bash
python 'scripts/01 inspect_columns.py'
```

`outputs/columns_2020.csv` ~ `outputs/columns_2025.csv`를 보고 `config/variable_mapping.yaml`을 코드북 기준으로 수정합니다.

```bash
python 'scripts/02 build_common_dataset.py'
python 'scripts/03 make_model_datasets.py'
python 'scripts/04 train_baseline_models.py'
```

`scripts/04 train_baseline_models.py`는 `비만`, `당뇨`, `고혈압` 데이터셋마다 RandomForest, XGBoost, LightGBM 분류 모델을 각각 학습합니다. 모델 파일은 `joblib` 압축으로 저장하고, Accuracy, Precision, Recall, F1, ROC-AUC를 비교한 포트폴리오용 리포트를 함께 생성합니다.

생성되는 주요 비교 산출물:

- `outputs/model_reports/model_comparison.csv`
- `outputs/model_reports/model_comparison.html`
- `outputs/model_reports/model_comparison.png`
- `outputs/model_reports/{질환}_{알고리즘}_report.json`
- `models/{질환}_{알고리즘}.joblib`

```bash
python 'scripts/05 inspect_model_sizes.py'
```

Flutter 앱 내부 추론용 ONNX 모델 생성:

```bash
python 'scripts/06 export_onnx_models.py'
```

`scripts/06 export_onnx_models.py`는 `outputs/model_reports/model_comparison.csv`에서 질환별 최고 모델을 선택해 ONNX로 변환합니다. 변환 뒤 `onnxruntime`으로 joblib 모델 예측값과 ONNX 예측값의 차이를 검증하고, Flutter 앱의 `assets/models/`에도 복사합니다.

## 중요 원칙

- 원자료 전체를 무작정 쓰지 않고, 앱 설문에서 물어볼 수 있는 변수만 사용합니다.
- `7`, `8`, `9`, `77`, `88`, `99`, `77777`, `88888`, `99999` 같은 특수값은 전체 일괄 치환하지 않습니다.
- 변수별 결측/범위/라벨 변환 규칙은 `config/variable_mapping.yaml`에서 관리합니다.
- 당뇨 모델 feature에는 `dia_*` 계열 진단/치료 변수를 넣지 않습니다.
- 고혈압 모델 feature에는 `hya_*` 계열 진단/치료 변수를 넣지 않습니다.
- 비만 모델 feature에는 `bmi`, `height`, `weight`를 넣지 않습니다. 이 값들은 label 생성에 사용되기 때문입니다.

## 산출물

- `Data/processed/chs2020_clean.csv` ~ `Data/processed/chs2025_clean.csv`
- `Data/ml/risk_common_2020_2025.csv`
- `Data/ml/obesity_dataset.csv`
- `Data/ml/diabetes_dataset.csv`
- `Data/ml/hypertension_dataset.csv`
- `outputs/data_quality_report.md`
- `outputs/model_reports/*.json`
- `outputs/model_reports/model_comparison.csv`
- `outputs/model_reports/model_comparison.html`
- `outputs/model_reports/model_comparison.png`
- `outputs/model_reports/onnx_export_report.json`
- `models/*.joblib`
- `models/onnx/*.onnx`
- `models/onnx/*_schema.json`

## FastAPI 예측 서버

설문조사 앱에서 JSON으로 응답값을 보내면 `models/*.joblib` 모델을 사용해 위험도 예측값을 반환합니다. 기본값은 `outputs/model_reports/model_comparison.csv`에서 ROC-AUC와 F1이 가장 높은 알고리즘을 자동 선택합니다. 특정 알고리즘을 지정하려면 `algorithm=random_forest`, `algorithm=xgboost`, `algorithm=lightgbm` 쿼리 파라미터를 사용합니다.

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

사용 가능한 모델과 feature 확인:

```bash
curl http://127.0.0.1:8000/models
```

단일 모델 예측:

```bash
curl -X POST http://127.0.0.1:8000/predict/obesity \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "smoking": 2,
    "drinking": 1,
    "physical_activity": 3,
    "walking": 4,
    "sleep_avg": 7,
    "stress": 2,
    "subjective_health": 3
  }'
```

XGBoost 모델로 단일 모델 예측:

```bash
curl -X POST 'http://127.0.0.1:8000/predict/obesity?algorithm=xgboost' \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "smoking": 2,
    "drinking": 1,
    "physical_activity": 3,
    "walking": 4,
    "sleep_avg": 7,
    "stress": 2,
    "subjective_health": 3
  }'
```

전체 모델 예측:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "bmi": 24.3,
    "smoking": 2,
    "drinking": 1,
    "physical_activity": 3,
    "walking": 4,
    "sleep_avg": 7,
    "stress": 2,
    "subjective_health": 3
  }'
```

개선 제안:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "sex": 1,
    "bmi": 31,
    "smoking": 1,
    "drinking": 1,
    "physical_activity": 0,
    "walking": 0,
    "sleep_avg": 6,
    "stress": 4,
    "subjective_health": 4
  }'
```

추천 API는 현재 입력값에서 BMI, 수면, 흡연, 음주, 운동, 걷기, 스트레스 응답을 바꾼 여러 시나리오를 재예측한 뒤, 위험도가 얼마나 낮아지는지 `improvement_percent_points`로 반환합니다. 범주형 코드가 의미하는 실제 문항 문구는 설문 앱의 코드북이 확정된 뒤 연결해야 합니다.

## 메모

`build_common_dataset.py`는 `Data/processed/*_risk.csv`가 있으면 이를 우선 사용합니다. 없으면 `config/variable_mapping.yaml`의 `raw` 경로에 있는 SAS 원자료를 읽습니다.

2023년은 수면시간 변수(`mtc_17z1`, `mtc_18z1`)가 없어 `sleep_avg`가 NaN으로 생성됩니다.
