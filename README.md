# Text-to-3D Hybrid Pipeline

**(Shap-E + DreamFusion 기반 하이브리드 3D 생성)**

---

## Project Summary

본 프로젝트는 Text-to-3D 생성에서 발생하는
**생성 속도와 기하 품질 간의 trade-off**를 완화하기 위해,
Shap-E와 DreamFusion을 결합한 **2-stage 하이브리드 파이프라인**을 구현한다.

Shap-E를 통해 텍스트로부터 **빠르게 초기 3D 메쉬(coarse geometry)**를 생성하고,
이를 DreamFusion의 geometry prior로 활용하여
고품질 메쉬를 보다 안정적으로 생성하는 것을 목표로 한다.

---

## Purpose

* 기존 Text-to-3D 모델의 한계 분석

  * Feed-forward 방식: 빠르나 기하 품질이 낮음
  * Optimization 기반 방식: 고품질이나 초기 수렴이 불안정
* Shap-E 기반 초기 기하 구조 활용
* DreamFusion의 SDS 최적화 효율 및 안정성 개선
* 최종 결과물을 **Unreal / Unity에서 활용 가능한 3D 에셋**으로 생성

---

## Overall Pipeline

1. 텍스트 프롬프트 입력
2. Shap-E를 이용한 초기 3D 메쉬 생성
3. 메쉬 정규화 및 좌표계 보정
4. 초기 메쉬를 density volume 형태로 변환
5. DreamFusion의 geometry prior로 입력
6. SDS 기반 최적화 수행
7. 최종 메쉬 추출 및 엔진 적용

---

## Code Structure

```
.
├── text_to_3d.py        # Shap-E 기반 텍스트 → 초기 3D 메쉬 생성
├── normalize_mesh.py   # 생성된 메쉬 정규화 (center, scale, axis)
├── main.py              # DreamFusion 학습 및 전체 파이프라인 실행
├── network.py           # NeRF 기반 geometry / color 네트워크 정의
├── network_grid.py      # Grid-based NeRF backbone
├── renderer.py          # Volume rendering 및 ray marching
├── utils.py             # 학습, loss, 카메라, 보조 유틸 함수
```

---

## Code Instruction

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

---

### 2. Initial Mesh Generation (Shap-E)

```bash
python text_to_3d.py
```

* 텍스트 프롬프트 입력
* Shap-E를 통해 초기 3D 메쉬(.obj) 생성

---

### 3. Mesh Preprocessing

```bash
python normalize_mesh.py --input input.obj --output normalized.obj
```

* 메쉬 중심 정렬
* 스케일 정규화
* 좌표계 보정 (Z-up → Y-up)

---

### 4. DreamFusion Optimization

```bash
python main.py --init_mesh normalized.obj
```

* 초기 메쉬를 density volume으로 변환
* DreamFusion의 random initialization을 대체
* SDS 기반 최적화 수행

---

## Model Details

### Geometry Prior Injection

* Shap-E 메쉬를 voxelized density grid로 변환
* `network.py`에서 초기 sigma 값으로 사용
* 불필요한 공간 탐색 감소

### Loss Design

* SDS (Score Distillation Sampling)
* Mask-weighted SDS
* Seam continuity loss
* Normal refinement loss
* Entropy / regularization loss

---

## Results

* DreamFusion 단독 대비:

  * 초기 수렴 안정성 개선
  * 구조 붕괴 감소
* 의미적으로 일관된 기하 구조 유지
* Unreal Engine에서 정상 렌더링 확인

---

## Demo

* Blender에서 텍스처 적용
* Unreal Engine에 직접 임포트하여 시연
* 게임 엔진 에셋으로 활용 가능성 확인

---

## Conclusion and Future Work

### Conclusion

* Shap-E + DreamFusion의 역할 분리형 파이프라인 제안
* 초기 메쉬 기반 geometry prior의 효과 확인
* Text-to-3D 생성 품질 및 실용성 개선

### Future Work

* 텍스처 베이킹 자동화
* 정량적 평가(CLIP, retrieval metric) 추가
* 실시간 생성 파이프라인 확장
