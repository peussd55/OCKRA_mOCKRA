# OCKRA & m-OCKRA 재현 실험

국제 저널에 게재된 **OCKRA**(One-Class K-means with Randomly-projected features Algorithm)과 그 확장판인 **m-OCKRA**(weighted attribute selection 기반 개선 모델)를 재현하기 위한 실험 노트북을 정리한 저장소입니다. 웨어러블 센서에서 수집된 단일 클래스 정상 활동 데이터를 기반으로 이상 징후를 탐지하는 절차를 단계별로 구성했으며, 연구 노트북과 발표 자료를 통해 실험 맥락을 손쉽게 파악할 수 있도록 구성했습니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L16-L84】【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L89-L216】

---

## 1. 저장소 구성
| 파일 | 설명 |
| --- | --- |
| `OCKRA.ipynb` | OCKRA 원 논문 절차를 기반으로 한 5-Fold 실험, 파라미터 탐색, 통계 검정을 포함합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L122-L212】【F:OCKRA_mOCKRA/OCKRA.ipynb†L288-L317】 |
| `m-OCKRA.ipynb` | m-OCKRA 알고리즘 재현을 위한 데이터 전처리, 다중 특징 선택, 가중치 앙상블, 대표 객체(MROs) 추출, Chebyshev 유사도 계산, 최종 성능 평가 절차를 구현합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L16-L669】【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L685-L2805】 |

---

## 2. 데이터셋 요약
- **센서 항목**: 자이로/가속도 평균·표준편차, 심박, 피부온도, 이동 관련 지표 등 26개 연속형 피처를 사용합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L31-L38】
- **원천 데이터**: `Dataset/NCDS.csv`(정상)와 `Dataset/ACDS.csv`(이상) 파일을 결합하여 FS-NCD, FS-NACD 두 가지 전처리 데이터셋을 생성합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L41-L55】
- **교차 검증 폴더 구조**: `FiveFoldCrossValidation/Fold{1..5}` 아래 `training.csv`, `testing.csv`를 사용하며, m-OCKRA 단계에서는 폴더 내부에 `projection/`, `MROs/`, `SimilarityResults_/`, `FinalResults_` 하위 폴더가 순차적으로 생성됩니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L152-L195】【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L500-L868】

> ⚠️ **경로 수정**: 노트북에는 로컬 Windows 경로가 하드코딩되어 있으므로, 실험 환경에 맞춰 절대/상대 경로를 재설정해야 합니다.

---

## 3. OCKRA 실험 파이프라인
1. **모델 정의**: 임의 특성 하위집합으로 `KMeans`를 학습하는 100개 이하의 분류기를 구성하고, 클러스터 중심까지의 거리로 유사도 점수를 계산합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L16-L84】
2. **데이터 전처리**: 60초 간격으로 샘플을 추출하여 시간 축 불균형을 완화한 뒤 `ANOM_COND` 라벨을 분리합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L130-L186】
3. **5-Fold 학습 & 저장**: 각 폴드마다 모델을 훈련한 뒤 `ockra_model_fold{n}.pkl`로 직렬화합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L148-L172】
4. **평가 및 로그**: 유사도 임계값 0.6을 사용해 정확도, 정밀도, 재현율, F1, AUC를 계산하고 폴드별 결과를 누적합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L104-L203】
5. **하이퍼파라미터 탐색**: `n_classifiers ∈ {25, 50, 100}`, `n_clusters ∈ {5, …, 50}` 조합을 탐색하고 결과를 `parameter_search_results.csv`에 저장합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L122-L212】
6. **통계 검정**: Friedman 검정을 통해 분류기 수와 클러스터 수가 AUC에 미치는 통계적 유의성을 파악합니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L288-L317】
7. **시각화**: 선택한 두 피처에 대해 K-means++ 클러스터링 결과를 확인할 수 있습니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L221-L263】

---

## 4. m-OCKRA 실험 파이프라인
1. **데이터 전처리**: 정상/이상 데이터를 통합하여 FS-NCD, FS-NACD 세트를 만들고 결측치를 제거합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L16-L55】
2. **개별 특징 선택**: Information Score(IS), Pearson Correlation(PC), Intra-Class Distance(ICD), Interquartile Range(IQR) 네 가지 기준으로 특징 점수 및 순위를 산출합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L89-L188】
3. **가중치 집계**: 평균, 중앙값, 최빈값, Borda Count를 이용해 다중 특징 선택 결과를 통합합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L232-L314】
4. **속성 투영 앙상블**: 각 가중치 컬럼과 속성 수(26→10) 조합마다 확률적 샘플링을 50회 반복해 가장 자주 선택된 피처를 투영 데이터셋으로 저장합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L483-L562】
5. **대표 객체(MROs) 추출**: RandomMiner 전략으로 부트스트랩 샘플에서 대표 객체를 선정하고 조건별 CSV로 저장합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L582-L669】
6. **Chebyshev 유사도 계산**: Numba로 가속한 Chebyshev 거리 기반 유사도 함수를 구현하고, 집계 가중치 상위 특성으로 동적 임계치(δ)를 계산해 폴드별 유사도를 산출합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L685-L835】
7. **폴드 평균화**: Fold별 유사도 파일을 결합해 라벨과 평균 유사도를 담은 결과 CSV를 생성합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L820-L870】
8. **최종 성능 평가**: 모든 조합의 평균 유사도 파일을 스캔하여 정확도, 정밀도, 재현율, F1, AUC를 계산하고, 최고 AUC 조합의 ROC 곡선을 시각화합니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L2720-L2804】

---

## 5. 실행 방법
1. **환경 준비**
   ```bash
   conda create -n ockra python=3.10
   conda activate ockra
   pip install pandas numpy scikit-learn scipy matplotlib numba joblib notebook
   ```
2. **데이터 배치**
   - `Dataset/`, `FiveFoldCrossValidation/` 폴더 구조를 원본 논문과 동일하게 맞춰 두 CSV를 배치합니다.
   - m-OCKRA 단계에서 생성되는 `feature_selection/`, `projection/`, `MROs/` 등의 하위 디렉터리는 노트북 실행 시 자동으로 생성됩니다.
3. **노트북 실행 순서**
   1. `m-OCKRA.ipynb`의 *Feature Selection1~3* 셀을 실행해 가중치 파일을 준비합니다.
   2. 동일 노트북에서 **속성 투영 → MROs → Chebyshev → 평균화 → 성능 평가** 셀을 순차 실행합니다.
   3. `OCKRA.ipynb`를 열어 5-Fold 학습, 파라미터 탐색, 통계 검정을 수행합니다.
4. **결과 확인**
   - `parameter_search_results.csv`, `FS_*_Aggregated_Results.csv`, `Final_Results_Summary.csv` 등의 산출물을 활용해 보고서 또는 발표 자료에 필요한 수치를 정리합니다.

---

## 6. 결과 요약 및 활용 팁
- **파라미터 탐색 결과**: 각 조합의 평균 성능이 `parameter_search_results.csv`에 기록되며, Friedman 검정 결과를 통해 조합 간 유의미한 차이를 검토할 수 있습니다.【F:OCKRA_mOCKRA/OCKRA.ipynb†L205-L317】
- **최적 조합 탐색**: m-OCKRA 실험에서는 `Final_Results_Summary.csv`에서 최고 AUC 조합과 ROC 곡선을 바로 확인할 수 있습니다.【F:OCKRA_mOCKRA/m-OCKRA.ipynb†L2720-L2804】

---

## 7. 향후 개선 아이디어
- 속성 수/대표 객체 비율 조합이 방대하므로, **병렬 처리** 또는 **샘플링 전략**을 도입해 실험 시간을 단축할 수 있습니다.
- Chebyshev 임계치 계산 시 사용한 단순 합 대신, **정규화된 가중치 평균**이나 **Bayesian 최적화**를 적용하면 보다 안정적인 기준값을 얻을 수 있습니다.
- `parameter_search_results.csv`와 `Final_Results_Summary.csv`를 통합 분석하여 **통합 하이퍼파라미터 추천 대시보드**를 구성해 보세요.

<img width="1152" height="515" alt="화면 캡처 2025-10-18 201157" src="https://github.com/user-attachments/assets/589343d8-9dd7-4824-b3f8-7889c9fa7d53" />
<img width="1143" height="512" alt="화면 캡처 2025-10-18 201224" src="https://github.com/user-attachments/assets/26d3f5f0-4965-4b0f-b924-c59a2a9dfc2b" />  
<img width="1147" height="526" alt="화면 캡처 2025-10-18 201407" src="https://github.com/user-attachments/assets/4405cf6f-8522-4da6-965b-d698aaec9480" />


