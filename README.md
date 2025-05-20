# Statistical-Analysis
Tuckey, Fisher, Duncan analysis
# 식품공학 연구를 위한 통계 분석 앱

이 Streamlit 앱은 식품공학 연구에서 자주 사용되는 통계 분석을 쉽게 수행할 수 있도록 개발되었습니다. ANOVA 분석과 함께 Tukey, Duncan, Fisher와 같은 다양한 Post-hoc 검정을 수행합니다.

## 기능

- 데이터 업로드 및 검증
- ANOVA 분석
- 다양한 Post-hoc 분석 (Tukey, Duncan, Fisher)
- 결과 시각화 (막대 그래프, 박스플롯, 바이올린 플롯)
- 결과 다운로드 (CSV, Excel)

## 설치 방법

```bash
# 저장소 복제
git clone https://github.com/[사용자명]/food-science-statistics.git
cd food-science-statistics

# 필요한 패키지 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py
