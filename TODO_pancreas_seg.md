# Pancreas Segmentation TODO

## 1. 다운로드 확인
- [ ] `images/` 2,238개, `labels/manual/` 482개, `labels/automatic/` 1,756개인지 확인
- [ ] image와 label 파일명이 `{pid}_{sid}`로 짝이 맞는지 확인

## 2. 데이터 현황 파악
- [ ] `clinical_information.csv`에서 `annotation_tier`(manual/automatic) × `label`(PDAC/non-PDAC) 교차표 작성
- [ ] `level` 열로 MSD 194건, NIH 80건 표시
- [ ] 같은 환자에 study가 여러 개인지(`patient_id` 중복) 확인

## 3. Label 기준 정하기
- [ ] 6-class 구성(0 background, 1 PDAC lesion, 2 veins, 3 arteries, 4 parenchyma, 5 duct, 6 CBD)을 실제 파일에서 확인
- [ ] 비교·학습에 쓸 "췌장" 정의 결정 (예: parenchyma(4) + lesion(1), 필요시 duct(5) 포함)

## 4. 데이터 분할
- [ ] `patient_id` 단위로 분할
- [ ] manual tier 일부를 test로 고정
- [ ] train = manual 나머지 + non-PDAC 케이스
- [ ] automatic PDAC 케이스를 train에 넣을지 결정

## 5. 목표 1: TotalSegmentator 확인
- [ ] manual tier에 TotalSegmentator(`--roi_subset pancreas`) 실행
- [ ] 전문가 lesion 영역 중 TotalSegmentator pancreas mask 안에 들어가는 비율 계산
- [ ] 결과에 따라 목표 2 진행 여부 결정

## 6. 목표 2: nnU-Net 학습 (목표 1에서 종양이 빠지는 경우)
- [ ] 4번 분할대로 nnU-Net 형식(`imagesTr/labelsTr`, `dataset.json`)으로 변환
- [ ] 학습
- [ ] (선택) manual-only 버전 vs manual + automatic PDAC 버전 비교
- [ ] test set(manual)에서 lesion Dice 계산 (parenchyma는 참고값)

## 7. 목표 3: 회전 MIP 투영 검증
- [ ] 3D 정답 mask와 예측 mask를 같은 각도로 회전·투영
- [ ] 각도별 2D Dice 계산
