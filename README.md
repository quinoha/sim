# QEDA: QEC Test Vector Generator & Evaluation Pipeline

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Tests](https://img.shields.io/badge/pytest-54%20passed-brightgreen.svg)
![Architecture](https://img.shields.io/badge/QEDA-Core%204--Stage-orange.svg)

**Built on Stim, Sinter, PyMatching, and stimbposd.**

## 🎯 Purpose & Architecture Overview

QEDA는 **양자 오류 정정(QEC) 시뮬레이션, 하드웨어(FPGA/ASIC/RTL) 테스트벡터 생성, 그리고 디코더 벤치마크 평가를 위한 종합 파이프라인**입니다.

QEDA Core 규범(Baseline 1.0)에 따라 **4대 Core Stage 단방향 파이프라인**으로 구성되어 있습니다:

```text
[Stage 1: SetupBuilder]  ──>  [Stage 2: EvaluationRunner]  ──>  [Stage 3: AnalysisEngine]  ──>  [Stage 4: ReportPublisher]
(m0_config & main.py)        (runner/ & Sinter/Stim)          (Planned: LER/CI/Pareto)       (Planned: 22 Figures PDF/HTML)
      │                                    │
      ▼                                    ▼
📦 EvaluationPlan                   📦 EvidenceBundle
(24개 불변 RunSpec)                 (.dets.b8, .obs.b8, manifest.json)
```

---

## ⚡ Key Features

1. **Stage 1 SetupBuilder (`m0_config/`)**:
   - 사람이 작성한 `configs/experiment.yaml`을 읽어 Pydantic v2 불변 모델로 0.1초 사전 검증 (Fail-Fast).
   - 시간/물리 단위 정규화 ($\text{ms} \to \text{ns}$, 비율 $\to$ fraction).
   - 직교 곱집합(Cartesian Product) 전개 및 비호환 조합(`bb_d6 × pymatching` 8개) Deny 필터링 $\to$ **최종 24개 Active `RunSpec` 확정**.
   - `load_evaluation_plan()` 유니버설 로더로 JSON/YAML 무손실 복원.

2. **Stage 2 EvaluationRunner (`runner/`, `qec_dataset/circuit_builder.py`)**:
   - **자동 회로 조립 브릿지**: `RunSpec` 사양을 읽어 Surface Code 및 Bivariate Bicycle(BB) Code의 `stim.Circuit`을 0.01초 내 자동 조립 (인메모리 스마트 캐싱).
   - **완전 디커플링 단방향 파이프라인**: 샷을 이중으로 뽑는 낭비 없이, Sinter/Stim으로 샷을 딱 1번만 채취하여 디스크에 `.dets.b8`(신드롬 문제지)과 `.obs.b8`(정답지)로 스트리밍 저장.
   - **순수 디코딩 지연시간(Pure Latency) 정밀 프로파일링**: 파이썬 프로세스 스폰 오버헤드 없이, 메모리에 로드된 신드롬 배열을 디코더(`decode_batch`)가 푸는 순간만 스톱워치로 정밀 측정 ($\mu\text{s}$ / shot).
   - **확장형 디코더 레지스트리**: PyMatching, BP-OSD뿐 아니라 사용자 커스텀 디코더(GNN `astra`, RTL 시뮬레이터)를 `register_decoder()`로 즉시 연동.

3. **Golden-Answer & Directed Corner Cases**:
   - Stim이 주입한 노이즈로부터 참값(Ground Truth) 로지컬 에러를 직접 기록.
   - 3대 결정론적 코너 케이스 (`all_zero`, `full_logical_error`, `near_miss`) 지원.

---

## 💻 Installation

본 프로젝트는 conda `gnn` 환경에서 개발 및 검증되었습니다:

```bash
conda activate gnn
cd sim
pip install -r requirements.txt
```

**설치 패키지 목록 (`requirements.txt`)**:
- `pydantic>=2.7.0`: 불변 실행 계약 모델 (`RunSpec`, `EvaluationPlan`, `EvidenceBundle`)
- `pyyaml>=6.0.1`: 실험 설정 파싱
- `stim>=1.16.0`, `sinter>=1.16.0`: 양자 회로 시뮬레이션 및 샷 샘플링
- `pymatching>=2.2.0`, `stimbposd>=0.2.0`, `ldpc>=0.1.50`: MWPM 및 BP-OSD 참조 디코더
- `numpy>=1.26.0`, `scipy>=1.11.0`: 선형대수 및 통계
- `pytest>=8.0.0`, `pytest-cov>=4.1.0`: 단위 테스트

---

## 🚀 Quick Start (CLI Usage)

### 1. 1단계 실행 계획 생성 및 24개 작업 매트릭스 표 확인
```bash
python main.py
```
> `configs/experiment.yaml`을 읽어 검증, 정규화, Deny 필터링 후 24개 RunSpec 표를 터미널에 출력합니다.

### 2. 기계 판독용 JSON 파일로 계획 저장
```bash
python main.py -o outputs/evaluation_plan.json
```

### 3. 2단계 시뮬레이션 원스톱 실행 (초고속 스모크 테스트 모드) ⭐
```bash
python main.py --run --quick
```
> 24개 전체 작업에 대해 100샷씩 실시간 회로 빌드 $\to$ 바이너리 샷 저장 $\to$ C++ 병렬 디코딩을 수행하고, **순수 디코딩 시간(ms, $\mu\text{s}$/shot)과 LER(논리 에러율)** 을 실시간으로 출력합니다.

### 4. 2단계 전체 프로덕션 샷 실행 (200,000 샷)
```bash
python main.py --run --workers 8
```

---

## 📂 Package Layout

```text
sim/
├── main.py                      # 전체 파이프라인 CLI 메인 엔트리포인트 (Setup & Run)
├── configs/
│   ├── experiment.yaml          # 메인 실험 사양서 (코드, 디코더, Sweep, 합격 기준)
│   ├── test_surface_d3.yaml     # 단일 Surface 테스트 설정
│   └── test_bb72_bposd.yaml     # 단일 BB 테스트 설정
├── m0_config/                   # [Stage 1] 설정 파서 및 SetupBuilder 서브시스템
│   ├── schema.py                # Pydantic v2 불변 설정 스키마
│   ├── validator.py             # 5대 의미론적 규칙 사전 검증기
│   ├── unit_normalizer.py       # 물리 단위 정규화기 (ms -> ns, % -> ratio)
│   ├── setup_builder.py         # 직교 곱집합 전개 & Deny 필터링 (24 RunSpec 생성)
│   ├── plan_schema.py           # EvaluationPlan, RunSpec 불변 모델
│   └── plan_loader.py           # load_evaluation_plan() 유니버설 로더
├── runner/                      # [Stage 2] EvaluationRunner 실행 엔진
│   ├── evidence_schema.py       # RunEvidence, EvidenceBundle 출력 스키마
│   └── evaluation_runner.py     # 단방향 샷 샘플링, 순수 디코딩 프로파일러, 채점기
├── qec_dataset/                 # 양자 회로 및 데이터셋 생성 코어
│   ├── circuit_builder.py       # RunSpec -> stim.Circuit 자동 조립 브릿지 (스마트 캐싱)
│   ├── circuits.py              # 일반 CSS 신드롬 추출 회로 빌더 (CNOT 스케줄링)
│   ├── sampling.py              # 바이너리 샷 스트리밍 (.dets.b8, .obs.b8)
│   ├── reference_decode.py      # PyMatching 및 BP-OSD 참조 디코더 인터페이스
│   ├── corner_cases.py          # 3대 결정론적 코너 케이스 (all-zero, full-err, near-miss)
│   ├── noise.py                 # NoiseModel (균일 디폴라라이징 4종 레이트)
│   ├── css_code.py              # CSSCode 데이터 구조 (Hx, Hz, Lx, Lz)
│   ├── gf2.py                   # GF(2) 갈루아 유한체 선형대수 (rank, nullspace)
│   ├── codes/
│   │   ├── surface.py           # Rotated Surface Code 생성기
│   │   └── bb.py                # Bivariate Bicycle Code 다항식 생성기 및 프리셋
│   ├── code_capacity.py         # GNN 학습용 코드 캐패시티 단일 라운드 데이터셋
│   └── astra_adapter.py         # Astra GNN 디코더 파이프라인 연동 뷰
├── outputs/                     # 산출물 디렉터리
│   ├── evaluation_plan.json     # 1단계 불변 실행 계약서
│   └── evidence/                # 2단계 바이너리 샷(.b8) 및 manifest.json, evidence_bundle.json
└── tests/                       # 종합 단위 테스트 스위트 (54개 테스트)
```

---

## 🔬 Python API Examples

### 1. EvaluationPlan 로드 및 stim.Circuit 자동 조립
```python
from m0_config import load_evaluation_plan
from qec_dataset import build_circuit_from_run_spec, build_all_circuits_from_plan

# 1. 계획서 로드
plan = load_evaluation_plan("outputs/evaluation_plan.json")

# 2. 첫 번째 RunSpec의 회로 즉시 생성
circuit = build_circuit_from_run_spec(plan.run_specs[0])
print(f"Qubits: {circuit.num_qubits}, Detectors: {circuit.num_detectors}")

# 3. 24개 회로 전체 일괄 빌드 (스마트 캐싱 적용으로 0.01초 소요)
all_circuits = build_all_circuits_from_plan(plan)
```

### 2. 2단계 시뮬레이션 및 순수 디코딩 실행
```python
from runner import EvaluationRunner

runner = EvaluationRunner(output_root="outputs/evidence")

# 24개 작업 전체 실행 (quick=True: 100샷 스모크 테스트)
bundle = runner.run_plan(plan, quick=True, quick_shots=100)

for ev in bundle.run_evidences[:3]:
    print(f"[{ev.code_id} x {ev.decoder_id}] LER: {ev.logical_error_rate:.3e}, Pure Latency: {ev.avg_latency_per_shot_ns / 1e3:.2f} us/shot")
```

### 3. 커스텀 디코더(GNN / RTL) 등록 및 실행
```python
import numpy as np

# 사용자 커스텀 디코더: 신드롬 행렬을 받아 예측 행렬 반환
def my_custom_gnn(syndromes: np.ndarray) -> np.ndarray:
    # 딥러닝 배치 추론 또는 C++ 가속기 연동
    return np.zeros((syndromes.shape[0], 1), dtype=np.uint8)

runner.register_decoder("my_gnn", my_custom_gnn)
```

---

## 🧪 Testing

전체 단위 테스트 실행:

```bash
pytest tests/ -v
```

**테스트 결과**: **`54 passed, 5 skipped in ~5s`** ($100\%$ 올패스)
- `test_m0_loader.py`: YAML 설정 파서 및 검증기
- `test_setup_builder.py`: 24개 RunSpec 전개 및 Deny 규칙
- `test_plan_loader.py`: 유니버설 EvaluationPlan 로더
- `test_circuit_builder.py`: RunSpec $\to$ Stim 양자 회로 자동 빌더
- `test_evaluation_runner.py`: 단방향 샷 채취 및 순수 디코딩 지연시간 측정
- `test_surface.py`, `test_bb.py`, `test_sampling.py`, `test_reference_decode.py`, `test_corner_cases.py`: 핵심 QEC 엔진
