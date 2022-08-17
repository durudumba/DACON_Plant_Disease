## DACON 작물 병해 진단 AI 경진대회

대회안내 : [DACON 농업 환경 변화에 따른 작물 병해 진단 AI경진대회][1]

[1]:https://dacon.io/competitions/official/235870/overview/description



#### 사전처리

------

image : 90º별로 회전, 3개로 자르기 (증강 데이터는 train과 같은 경로에 저장)

tabular : 이틀 동안의 환경 변화가 작물 병해 발생이나 진행여부에 영향을 미치지 않을 것으로 판단, 이미지만으로 잘 분류하지 못하는 품종의 환경데이터를 비교해본 결과 온도, 이슬점, CO2 데이터의 최소, 최고, 평균값들의 산술평균



#### 학습

------

* 단일 사전학습 모델(EfficientNet-b0)
* 각 레이블(작물-병해-진행도)에서 최소갯수 레이블의 갯수만큼 샘플링
* 10 epoch  / 15 sample at each epoch



#### 성적

------
F1-score : 0.94102(73th)

