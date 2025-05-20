# pip install streamlit pandas scipy statsmodels matplotlib openpyxl seaborn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Jong Ho Lee's Statistical Analysis", layout="wide")
st.title("Jong Ho Lee's Statistical Analysis")

# 사이드바 리셋 버튼
if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.rerun()

# 데이터 업로드 및 분석 방법 선택
st.subheader("1️⃣ 데이터 업로드")
uploaded_file = st.file_uploader("📁 Excel 파일 업로드 (1열: 그룹, 2열: 값)", type=["xlsx", "xls"])

st.subheader("2️⃣ Post-hoc 분석 방법 선택")
methods = st.multiselect(
    "원하는 사후 분석 방법을 선택하세요",
    ['Tukey', 'Duncan', 'Fisher'],
    default=['Tukey']
)

# 시각화 옵션
viz_type = st.radio(
    "시각화 유형",
    ["막대 그래프", "박스플롯", "바이올린 플롯"],
    horizontal=True
)

# Fisher LSD 구현
def perform_fisher(df, alpha=0.05):
    model = ols('Value ~ C(Group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
    df_resid = anova_table.loc["Residual", "df"]
    
    # 모든 그룹 쌍별 비교
    fisher_results = []
    group_means = df.groupby('Group')['Value'].mean()
    group_counts = df.groupby('Group')['Value'].count()
    groups = group_means.index.tolist()
    
    for i, group1 in enumerate(groups):
        for j in range(i+1, len(groups)):
            group2 = groups[j]
            mean_diff = abs(group_means[group1] - group_means[group2])
            
            # 표준 오차 계산
            se = np.sqrt(mse * (1/group_counts[group1] + 1/group_counts[group2]))
            t_val = mean_diff / se
            p_val = 2 * (1 - stats.t.cdf(abs(t_val), df_resid))
            
            fisher_results.append({
                'Group1': group1,
                'Group2': group2,
                'Mean Diff': mean_diff,
                'Std Error': se,
                't-value': t_val,
                'p-value': p_val,
                'Significant': p_val < alpha
            })
    
    # 그룹 라벨링
    sorted_groups = group_means.sort_values(ascending=False).index.tolist()
    labels = {sorted_groups[0]: 'a'}
    
    for i in range(1, len(sorted_groups)):
        curr_group = sorted_groups[i]
        prev_group = sorted_groups[i-1]
        
        # 이전 그룹과 유의적 차이가 있는지 확인
        is_significant = False
        for result in fisher_results:
            if ((result['Group1'] == curr_group and result['Group2'] == prev_group) or 
                (result['Group1'] == prev_group and result['Group2'] == curr_group)):
                is_significant = result['Significant']
                break
        
        if is_significant:
            # 유의적 차이가 있으면 새 라벨
            labels[curr_group] = chr(ord(labels[prev_group]) + 1)
        else:
            # 유의적 차이가 없으면 이전 라벨
            labels[curr_group] = labels[prev_group]
    
    # 결과를 DataFrame으로 반환
    results_df = pd.DataFrame(fisher_results)
    labels_df = pd.DataFrame({
        'Group': list(labels.keys()),
        'Fisher_Label': list(labels.values())
    })
    
    return results_df, labels_df

# Tukey HSD 구현 - 완전히 재작성
def perform_tukey(df, alpha=0.05):
    # 직접 모든 쌍의 비교 수행
    tukey_results = []
    
    # 그룹 통계
    group_means = df.groupby('Group')['Value'].mean()
    group_counts = df.groupby('Group')['Value'].count()
    groups = sorted(group_means.index.tolist())
    
    # ANOVA 모델 생성
    model = ols('Value ~ C(Group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # MSE 계산
    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
    df_resid = anova_table.loc["Residual", "df"]
    
    # q 임계값 계산 (studentized range statistic)
    k = len(groups)  # 그룹 수
    q_crit = stats.studentized_range.ppf(1 - alpha, k, df_resid)
    
    # 모든 가능한 쌍에 대해 Tukey HSD 계산
    for i, group1 in enumerate(groups):
        for j in range(i+1, len(groups)):
            group2 = groups[j]
            
            # 평균 차이
            mean_diff = group_means[group1] - group_means[group2]
            
            # 표준 오차
            n_harmonic = 2 / (1/group_counts[group1] + 1/group_counts[group2])
            se = np.sqrt(mse / n_harmonic)
            
            # q 통계량
            q_stat = abs(mean_diff) / se
            
            # 유의성 판단
            significant = q_stat > q_crit
            
            # 신뢰구간
            ci_lower = mean_diff - q_crit * se
            ci_upper = mean_diff + q_crit * se
            
            # p-value 근사값 (정확한 계산은 더 복잡함)
            p_val = 1 - stats.studentized_range.cdf(q_stat, k, df_resid)
            
            tukey_results.append({
                'Group1': group1,
                'Group2': group2,
                'Mean Diff': mean_diff,
                'Std Error': se, 
                'q-value': q_stat,
                'Lower CI': ci_lower,
                'Upper CI': ci_upper,
                'p-value': p_val,
                'Significant': significant
            })
    
    # 그룹 라벨링
    sorted_groups = group_means.sort_values(ascending=False).index.tolist()
    labels = {sorted_groups[0]: 'a'}
    
    for i in range(1, len(sorted_groups)):
        curr_group = sorted_groups[i]
        prev_group = sorted_groups[i-1]
        
        # 이전 그룹과 유의적 차이가 있는지 확인
        is_significant = False
        for result in tukey_results:
            if ((result['Group1'] == curr_group and result['Group2'] == prev_group) or 
                (result['Group1'] == prev_group and result['Group2'] == curr_group)):
                is_significant = result['Significant']
                break
        
        if is_significant:
            # 유의적 차이가 있으면 새 라벨
            labels[curr_group] = chr(ord(labels[prev_group]) + 1)
        else:
            # 유의적 차이가 없으면 이전 라벨
            labels[curr_group] = labels[prev_group]
    
    # 결과를 DataFrame으로 반환
    results_df = pd.DataFrame(tukey_results)
    labels_df = pd.DataFrame({
        'Group': list(labels.keys()),
        'Tukey_Label': list(labels.values())
    })
    
    return results_df, labels_df

# Duncan 방법 구현
def perform_duncan(df, alpha=0.05):
    model = ols('Value ~ C(Group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    mse = anova_table.loc["Residual", "sum_sq"] / anova_table.loc["Residual", "df"]
    df_resid = anova_table.loc["Residual", "df"]
    
    # 그룹 평균 및 정렬
    group_means = df.groupby('Group')['Value'].mean()
    sorted_groups = group_means.sort_values(ascending=False).index.tolist()
    group_counts = df.groupby('Group')['Value'].count()
    
    # Duncan 임계값 계산을 위한 간단한 함수
    def get_critical_range(r, df_resid, mse, n, alpha=0.05):
        # 간략화된 Duncan 테이블 (실제로는 더 자세한 테이블 필요)
        q_values = {
            2: 3.0, 3: 3.1, 4: 3.2, 5: 3.3, 6: 3.4, 7: 3.5
        }
        # 범위에 따른 q값 선택 (범위가 테이블을 벗어나면 최대값 사용)
        q = q_values.get(r, 3.6)
        return q * np.sqrt(mse / n)
    
    # 평균 샘플 크기 계산
    avg_n = np.mean(list(group_counts))
    
    # 모든 그룹 쌍별 비교
    duncan_results = []
    
    for i, group1 in enumerate(sorted_groups):
        for j in range(i+1, len(sorted_groups)):
            group2 = sorted_groups[j]
            mean_diff = abs(group_means[group1] - group_means[group2])
            
            # 범위에 따른 임계값 계산
            r = j - i + 1
            critical_range = get_critical_range(r, df_resid, mse, avg_n, alpha)
            is_significant = mean_diff > critical_range
            
            duncan_results.append({
                'Group1': group1,
                'Group2': group2,
                'Mean Diff': mean_diff,
                'Critical Range': critical_range,
                'Range Size': r,
                'Significant': is_significant
            })
    
    # 그룹 라벨링
    labels = {sorted_groups[0]: 'a'}
    
    for i in range(1, len(sorted_groups)):
        curr_group = sorted_groups[i]
        prev_group = sorted_groups[i-1]
        
        # 이전 그룹과 유의적 차이가 있는지 확인
        is_significant = False
        for result in duncan_results:
            if ((result['Group1'] == curr_group and result['Group2'] == prev_group) or 
                (result['Group1'] == prev_group and result['Group2'] == curr_group)):
                is_significant = result['Significant']
                break
        
        if is_significant:
            # 유의적 차이가 있으면 새 라벨
            labels[curr_group] = chr(ord(labels[prev_group]) + 1)
        else:
            # 유의적 차이가 없으면 이전 라벨
            labels[curr_group] = labels[prev_group]
    
    # 결과를 DataFrame으로 반환
    results_df = pd.DataFrame(duncan_results)
    labels_df = pd.DataFrame({
        'Group': list(labels.keys()),
        'Duncan_Label': list(labels.values())
    })
    
    return results_df, labels_df

# 데이터 검증 및 처리
def process_data(file):
    try:
        df = pd.read_excel(file, header=None)
        
        # 열 이름 설정
        if len(df.columns) >= 2:
            df.columns = ['Group', 'Value'] if len(df.columns) == 2 else df.columns.tolist()[:2] + ['Unnamed'] * (len(df.columns) - 2)
            df = df[['Group', 'Value']]  # 필요한 열만 선택
        else:
            st.error("❌ 파일에 최소 2개 이상의 열이 필요합니다.")
            return None
        
        # 데이터 타입 변환
        df['Group'] = df['Group'].astype(str)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        
        # 누락값 제거
        null_count = df['Value'].isna().sum()
        if null_count > 0:
            st.warning(f"⚠️ {null_count}개의 행에서 숫자로 변환할 수 없는 값이 발견되어 제외되었습니다.")
            df = df.dropna(subset=['Value'])
        
        if len(df) == 0:
            st.error("❌ 유효한 숫자 데이터가 없습니다.")
            return None
        
        # 그룹 수 확인
        if df['Group'].nunique() < 2:
            st.error("❌ ANOVA 분석을 위해 최소 2개 이상의 그룹이 필요합니다.")
            return None
        
        return df
    
    except Exception as e:
        st.error(f"❌ 파일 처리 중 오류 발생: {str(e)}")
        return None

# ANOVA 분석
def perform_anova(df):
    model = ols('Value ~ C(Group)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    f_stat = anova_table.loc['C(Group)', 'F']
    p_val = anova_table.loc['C(Group)', 'PR(>F)']
    
    # 등분산성 검정
    groups = [df[df['Group'] == group]['Value'] for group in df['Group'].unique()]
    levene_stat, levene_p = stats.levene(*groups)
    
    return {
        'f_stat': f_stat,
        'p_val': p_val,
        'levene_stat': levene_stat,
        'levene_p': levene_p,
        'anova_table': anova_table
    }

# 결과 시각화
def create_visualization(df, summary_df, viz_type):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 색상 설정
    colors = sns.color_palette("Set2", len(df['Group'].unique()))
    
    if viz_type == "막대 그래프":
        # 막대 그래프
        bars = ax.bar(summary_df['Group'], summary_df['Mean'], yerr=summary_df['Std'], 
                     capsize=5, color=colors)
        ax.set_ylabel("Mean Value")
        ax.set_title("Group Comparison (Mean ± SD)")
        
        # 라벨 위치 계산
        for i, row in summary_df.iterrows():
            y_pos = row['Mean'] + row['Std'] + 0.1
            
            # 라벨 텍스트 생성
            labels = []
            for method in methods:
                label_col = f'{method}_Label'
                if label_col in row and pd.notna(row[label_col]):
                    labels.append(row[label_col])
            
            if labels:
                # 모든 라벨 붙이기
                label_text = ','.join(labels)
                ax.text(i, y_pos, label_text, ha='center', va='bottom', 
                       fontsize=12, fontweight='bold', color='red')
    
    elif viz_type == "박스플롯":
        # 박스플롯
        sns.boxplot(x='Group', y='Value', data=df, ax=ax, palette="Set2")
        ax.set_title("Group Comparison (Boxplot)")
        
        # 그룹별 라벨 추가
        for i, group in enumerate(df['Group'].unique()):
            row = summary_df[summary_df['Group'] == group]
            if not row.empty:
                # 라벨 텍스트 생성
                labels = []
                for method in methods:
                    label_col = f'{method}_Label'
                    if label_col in row.columns and pd.notna(row[label_col].values[0]):
                        labels.append(row[label_col].values[0])
                
                if labels:
                    # 그룹별 최대값 위에 라벨 표시
                    y_max = df[df['Group'] == group]['Value'].max()
                    y_pos = y_max + 0.1
                    label_text = ','.join(labels)
                    ax.text(i, y_pos, label_text, ha='center', va='bottom', 
                           fontsize=12, fontweight='bold', color='red')
    
    elif viz_type == "바이올린 플롯":
        # 바이올린 플롯
        sns.violinplot(x='Group', y='Value', data=df, ax=ax, palette="Set2", inner='box')
        ax.set_title("Group Comparison (Violin Plot)")
        
        # 그룹별 라벨 추가 (박스플롯과 동일)
        for i, group in enumerate(df['Group'].unique()):
            row = summary_df[summary_df['Group'] == group]
            if not row.empty:
                # 라벨 텍스트 생성
                labels = []
                for method in methods:
                    label_col = f'{method}_Label'
                    if label_col in row.columns and pd.notna(row[label_col].values[0]):
                        labels.append(row[label_col].values[0])
                
                if labels:
                    # 그룹별 최대값 위에 라벨 표시
                    y_max = df[df['Group'] == group]['Value'].max()
                    y_pos = y_max + 0.1
                    label_text = ','.join(labels)
                    ax.text(i, y_pos, label_text, ha='center', va='bottom', 
                           fontsize=12, fontweight='bold', color='red')
    
    # 그래프 스타일 조정
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

# 메인 로직
if uploaded_file:
    # 1. 데이터 처리
    df = process_data(uploaded_file)
    
    if df is not None:
        # 2. 데이터 미리보기
        st.subheader("3️⃣ 데이터 미리보기")
        
        # 데이터 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 데이터 수", len(df))
        with col2:
            st.metric("그룹 수", df['Group'].nunique())
        with col3:
            st.metric("그룹별 평균 데이터 수", round(len(df) / df['Group'].nunique(), 1))
        
        # 데이터프레임 표시
        st.dataframe(df)
        
        # 3. ANOVA 분석
        anova_results = perform_anova(df)
        
        st.subheader("4️⃣ ANOVA 분석 결과")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("F 통계량", round(anova_results['f_stat'], 4))
            st.metric("p-값", round(anova_results['p_val'], 4))
        
        with col2:
            st.metric("Levene 통계량 (등분산성)", round(anova_results['levene_stat'], 4))
            st.metric("Levene p-값", round(anova_results['levene_p'], 4))
        
        alpha = 0.05  # 기본 유의수준
        
        # ANOVA 결과 해석
        if anova_results['p_val'] < alpha:
            st.success(f"✅ 그룹 간 통계적으로 유의한 차이가 있습니다 (p < {alpha}).")
            
            # 4. Post-hoc 분석 (ANOVA가 유의한 경우에만)
            if methods:
                st.subheader("5️⃣ Post-hoc 분석 결과")
                
                # 기본 통계 계산
                summary_df = df.groupby('Group').agg(
                    Mean=('Value', 'mean'),
                    Std=('Value', 'std'),
                    N=('Value', 'count'),
                    Min=('Value', 'min'),
                    Max=('Value', 'max')
                ).reset_index()
                
                # 각 방법별 분석 및 라벨링
                for method in methods:
                    st.markdown(f"**{method} 분석 결과:**")
                    
                    try:
                        if method == 'Tukey':
                            results_df, labels_df = perform_tukey(df, alpha)
                        elif method == 'Duncan':
                            results_df, labels_df = perform_duncan(df, alpha)
                        elif method == 'Fisher':
                            results_df, labels_df = perform_fisher(df, alpha)
                        
                        # 결과 표시
                        st.dataframe(results_df)
                        
                        # 라벨 병합
                        summary_df = pd.merge(summary_df, labels_df, on='Group', how='left')
                        
                    except Exception as e:
                        st.error(f"❌ {method} 분석 중 오류 발생: {str(e)}")
                
                # 5. 최종 요약 테이블
                st.subheader("6️⃣ 최종 요약 테이블")
                st.dataframe(summary_df)
                
                # 6. 시각화
                st.subheader("7️⃣ 결과 시각화")
                fig = create_visualization(df, summary_df, viz_type)
                st.pyplot(fig)
                
                # 7. 다운로드 버튼
                st.subheader("8️⃣ 결과 다운로드")
                
                # CSV 다운로드
                csv = summary_df.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                st.download_button(
                    label="⬇️ CSV로 다운로드",
                    data=csv,
                    file_name=f"statistical_analysis_{timestamp}.csv",
                    mime="text/csv"
                )
                
                # Excel 다운로드 (모든 결과 포함)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    pd.DataFrame(anova_results['anova_table']).to_excel(writer, sheet_name='ANOVA', index=True)
                    df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                excel_data = buffer.getvalue()
                
                st.download_button(
                    label="⬇️ Excel로 다운로드 (전체 결과)",
                    data=excel_data,
                    file_name=f"statistical_analysis_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            else:
                st.warning("사후 분석을 수행하려면 최소 하나 이상의 방법을 선택하세요.")
        else:
            st.info(f"ℹ️ 그룹 간 통계적으로 유의한 차이가 없습니다 (p > {alpha}). 사후 분석이 필요하지 않습니다.")
else:
    # 데이터 파일이 업로드되지 않은 경우
    st.info("👆 분석을 시작하려면 Excel 파일을 업로드해주세요.")
    
    # 예시 데이터 형식 표시
    with st.expander("📝 예시 데이터 형식", expanded=True):
        st.markdown("""
        ### 엑셀 파일 형식
        
        파일은 다음과 같은 구조여야 합니다:
        
        | 그룹 | 값 |
        |------|------|
        | A | 10.5 |
        | A | 11.2 |
        | A | 9.8 |
        | B | 12.3 |
        | B | 14.5 |
        | B | 13.2 |
        | C | 15.6 |
        | C | 16.2 |
        | C | 14.8 |
        
        * **첫 번째 열**: 그룹 이름 (문자열)
        * **두 번째 열**: 측정값 (숫자)
        * 헤더 행은 필요하지 않습니다
        """)
        
        # 예시 다운로드 제공
        st.markdown("### 예시 파일 다운로드")
        
        # 예시 데이터 생성
        example_data = pd.DataFrame({
            'Group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
            'Value': [10.5, 11.2, 9.8, 10.1, 12.3, 14.5, 13.2, 11.9, 15.6, 16.2, 14.8, 15.9]
        })
        
        # 예시 파일 다운로드 버튼
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            example_data.to_excel(writer, index=False, header=False)
        
        st.download_button(
            label="📥 예시 데이터 다운로드",
            data=buffer.getvalue(),
            file_name="example_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
