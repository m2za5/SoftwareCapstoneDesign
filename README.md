# Text-to-3D Hybrid Pipeline (Shap-E + DreamFusion)

본 프로젝트는 소프트웨어융합학과 캡스톤디자인 과제로 수행된 연구로,
Shap-E와 DreamFusion의 장점을 결합한 하이브리드 Text-to-3D 생성 파이프라인을 구현한다.

---

## Project Overview

- **Goal**: 생성 속도와 기하 품질을 동시에 고려한 Text-to-3D 파이프라인 구현
- **Approach**:
  - Shap-E를 이용한 빠른 초기 3D 기하 구조 생성
  - 초기 메쉬를 geometry prior로 활용하여 DreamFusion 최적화 수행
- **Output**: Unreal Engine에서 활용 가능한 3D 메쉬 결과

---

## Pipeline

1. Text Prompt 입력
2. Shap-E 기반 초기 3D 메쉬 생성
3. 메쉬 전처리 및 density-based geometry prior 변환
4. DreamFusion(SDS) 기반 기하 구조 정제
5. 최종 메쉬 추출 및 엔진 적용

---

## How to Run

```bash
# Step 1. Shap-E 초기 메쉬 생성
python scripts/run_shape_init.py --prompt "a rabbit"

# Step 2. DreamFusion 최적화
python scripts/run_dreamfusion.py --init_mesh path/to/mesh.obj
