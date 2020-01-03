# CRP(Crouse Register Prediction) Project
![](https://media0.giphy.com/media/Ly6FB6xRSJlW8/giphy.gif)

## 프로젝트 소개
 - 2019년 2학기에 했던 Ybigta 컨퍼런스 주제! <투표 1등>
 - 마일리지로 운영되는 수강신청 제도에서 본인의 스펙에 따라 해당 과목의 수강 성공여부를 예측해주고 나아가 최적의 마일리지를 계산해주는 ML 프로젝트!
 - ML 모델을 구축한 다음에 이를 웹으로 구축하는 것까지 시도했음!
 - 참여자: 안주영, 이용하, 정현우, 최민태, 한승희

## 프로젝트 목표
 1. 과목별 최적의 마일리지를 계산해 추천해준다.
 2. 이를 사용하기 편하게 웹으로 구현하여 가독성과 접근성을 높인다.
 3. 최적 마일리지 시스템을 통해서 성공적인 수강신청을 이룬다!
 
## 사용 기법들
 1. 크롤링: Selenium, BeatifulSoup
 2. DB구축: AWS, MySQL, pyMySql
 3. Machine learning: XGBoost, GridSearchCV
 4. Web: Flask, HTML, CSS, Javascript
 
## 한계점  
 1. 시계열 순서 고려: 컨퍼런스 당일날 나왔던 지적점으로 기존 수강신청 데이터를 바탕으로 예측을 하기 위해서는 시계열 순서를 고려했어야 되지 않냐고 물었었음. 
 2. 인터랙티브한 웹 구현에는 실패: Javascript은 다들 거의 처음 다뤄보아서 역동적인 Web을 구현하는 데에는 한계가 있었다. Javascript랑 HTML, CSS를 좀 더 공부하여 더 이쁜 >.< Web을 만들고 싶다!!!
