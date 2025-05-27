"""
RCBD (Randomized Complete Block Design) 통계 분석
프레이밍 효과와 블록 효과를 분석하기 위한 ANOVA 수행
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

class RCBDAnalyzer:
    def __init__(self):
        """RCBD 분석기 초기화"""
        pass
    
    def prepare_dataframe(self, experimental_data):
        """
        실험 데이터를 RCBD 분석을 위한 DataFrame으로 변환
        
        Args:
            experimental_data (list): 일관성 점수가 포함된 실험 데이터
            
        Returns:
            pd.DataFrame: RCBD 분석용 DataFrame
        """
        rows = []
        
        for data in experimental_data:
            if data.get('bert_multilingual_similarity') is not None:
                rows.append({
                    'category': data['category'],
                    'framing_level': data['framing_level'],
                    'base_question': data['base_question'],
                    'bert_multilingual_similarity': data['bert_multilingual_similarity'],
                    'n_responses': data.get('n_responses', 0)
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def perform_rcbd_analysis(self, df, response_var='bert_multilingual_similarity'):
        """
        RCBD ANOVA 분석 수행
        
        Args:
            df (pd.DataFrame): 분석용 데이터프레임
            response_var (str): 반응변수 ('bert_multilingual_similarity')
            
        Returns:
            dict: 분석 결과
        """
        print(f"\n{'='*60}")
        print(f"RCBD 분석 - 반응변수: {response_var}")
        print(f"{'='*60}")
        
        # 데이터 체크
        print(f"전체 관측치 수: {len(df)}")
        print(f"카테고리 수: {df['category'].nunique()}")
        print(f"프레이밍 수준 수: {df['framing_level'].nunique()}")
        
        # 각 조건별 샘플 수 확인
        print(f"\n조건별 샘플 수:")
        condition_counts = df.groupby(['category', 'framing_level']).size()
        print(condition_counts)
        
        # RCBD 모델 (처리: framing_level, 블록: category)
        formula = f"{response_var} ~ C(framing_level) + C(category)"
        model = ols(formula, data=df).fit()
        
        # ANOVA 수행
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        print(f"\n📊 RCBD ANOVA 결과 ({response_var})")
        print("-" * 50)
        print(anova_table)
        
        # 효과 크기 계산
        total_ss = anova_table['sum_sq'].sum()
        framing_eta_squared = anova_table.loc['C(framing_level)', 'sum_sq'] / total_ss
        category_eta_squared = anova_table.loc['C(category)', 'sum_sq'] / total_ss
        
        print(f"\n📈 효과 크기 (Eta-squared)")
        print("-" * 30)
        print(f"프레이밍 효과: {framing_eta_squared:.4f}")
        print(f"블록(카테고리) 효과: {category_eta_squared:.4f}")
        
        # 평균값 비교
        framing_means = df.groupby('framing_level')[response_var].agg(['mean', 'std', 'count'])
        category_means = df.groupby('category')[response_var].agg(['mean', 'std', 'count'])
        
        print(f"\n📋 프레이밍별 평균")
        print("-" * 40)
        for framing in framing_means.index:
            mean_val = framing_means.loc[framing, 'mean']
            std_val = framing_means.loc[framing, 'std']
            count_val = framing_means.loc[framing, 'count']
            print(f"{framing:>8}: {mean_val:.4f} (±{std_val:.4f}) [n={count_val}]")
        
        print(f"\n📋 카테고리별 평균")
        print("-" * 40)
        for category in category_means.index:
            mean_val = category_means.loc[category, 'mean']
            std_val = category_means.loc[category, 'std']
            count_val = category_means.loc[category, 'count']
            print(f"{category:>8}: {mean_val:.4f} (±{std_val:.4f}) [n={count_val}]")
        
        # 결과 딕셔너리 반환
        results = {
            'model': model,
            'anova_table': anova_table,
            'framing_means': framing_means,
            'category_means': category_means,
            'framing_eta_squared': framing_eta_squared,
            'category_eta_squared': category_eta_squared,
            'response_variable': response_var
        }
        
        return results
    
    def perform_full_factorial_analysis(self, df, response_var='bert_multilingual_similarity'):
        """
        전체 요인 설계 분석 (교호작용 포함)
        
        Args:
            df (pd.DataFrame): 분석용 데이터프레임
            response_var (str): 반응변수
            
        Returns:
            dict: 분석 결과
        """
        print(f"\n{'='*60}")
        print(f"전체 요인 설계 분석 - 반응변수: {response_var}")
        print(f"{'='*60}")
        
        try:
            # 교호작용을 포함한 모델
            formula = f"{response_var} ~ C(framing_level) * C(category)"
            model = ols(formula, data=df).fit()
            
            # ANOVA 수행
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            print(f"\n📊 전체 요인 ANOVA 결과 ({response_var})")
            print("-" * 50)
            print(anova_table)
            
            # 교호작용 평균 계산
            interaction_means = df.groupby(['framing_level', 'category'])[response_var].mean().unstack()
            
            print(f"\n📋 교호작용 평균")
            print("-" * 40)
            print(interaction_means)
            
            results = {
                'model': model,
                'anova_table': anova_table,
                'interaction_means': interaction_means,
                'response_variable': response_var
            }
            
        except Exception as e:
            print(f"\n⚠️  교호작용 분석 실패: {e}")
            print("   데이터가 부족하거나 수치적 문제가 발생했습니다.")
            print("   RCBD 기본 분석 결과를 사용하세요.")
            
            # 기본 교호작용 평균만 계산
            interaction_means = df.groupby(['framing_level', 'category'])[response_var].mean().unstack()
            
            print(f"\n📋 교호작용 평균 (기본)")
            print("-" * 40)
            print(interaction_means)
            
            results = {
                'model': None,
                'anova_table': None,
                'interaction_means': interaction_means,
                'response_variable': response_var,
                'error': str(e)
            }
        
        return results
    
    def plot_results(self, df, response_var='bert_multilingual_similarity', figsize=(15, 10)):
        """
        분석 결과 시각화
        
        Args:
            df (pd.DataFrame): 분석용 데이터프레임
            response_var (str): 반응변수
            figsize (tuple): 그래프 크기
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 프레이밍별 박스플롯
        sns.boxplot(data=df, x='framing_level', y=response_var, ax=axes[0,0])
        axes[0,0].set_title(f'프레이밍별 {response_var}')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. 카테고리별 박스플롯
        sns.boxplot(data=df, x='category', y=response_var, ax=axes[0,1])
        axes[0,1].set_title(f'카테고리별 {response_var}')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 교호작용 플롯
        sns.pointplot(data=df, x='framing_level', y=response_var, 
                     hue='category', ax=axes[1,0])
        axes[1,0].set_title(f'프레이밍 × 카테고리 교호작용')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 히트맵
        heatmap_data = df.groupby(['category', 'framing_level'])[response_var].mean().unstack()
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', ax=axes[1,1])
        axes[1,1].set_title(f'{response_var} 평균값 히트맵')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_comprehensive_report(self, experimental_data):
        """
        종합 분석 보고서 생성
        
        Args:
            experimental_data (list): 실험 데이터
            
        Returns:
            dict: 종합 분석 결과
        """
        # 데이터프레임 준비
        df = self.prepare_dataframe(experimental_data)
        
        print(f"\n🔬 RCBD 종합 분석 보고서")
        print(f"{'='*80}")
        
        # 기본 정보
        print(f"\n📋 실험 설계 정보")
        print(f"   - 전체 관측치: {len(df)}")
        print(f"   - 프레이밍 수준: {list(df['framing_level'].unique())}")
        print(f"   - 카테고리(블록): {list(df['category'].unique())}")
        
        # BERT Multilingual 유사도 분석
        bert_multilingual_results = self.perform_rcbd_analysis(df, 'bert_multilingual_similarity')
        
        # 전체 요인 분석 (교호작용 포함)
        bert_multilingual_factorial = self.perform_full_factorial_analysis(df, 'bert_multilingual_similarity')
        
        # 결과 종합
        report = {
            'dataframe': df,
            'bert_multilingual_rcbd': bert_multilingual_results,
            'bert_multilingual_factorial': bert_multilingual_factorial,
            'summary': self._generate_summary(bert_multilingual_results)
        }
        
        return report
    
    def _generate_summary(self, bert_multilingual_results):
        """
        분석 결과 요약 생성
        
        Args:
            bert_multilingual_results (dict): BERT Multilingual 유사도 분석 결과
            
        Returns:
            dict: 요약 정보
        """
        summary = {}
        
        # BERT Multilingual 유사도 요약
        bert_multilingual_anova = bert_multilingual_results['anova_table']
        summary['bert_multilingual_framing_pvalue'] = bert_multilingual_anova.loc['C(framing_level)', 'PR(>F)']
        summary['bert_multilingual_category_pvalue'] = bert_multilingual_anova.loc['C(category)', 'PR(>F)']
        summary['bert_multilingual_framing_significant'] = summary['bert_multilingual_framing_pvalue'] < 0.05
        summary['bert_multilingual_category_significant'] = summary['bert_multilingual_category_pvalue'] < 0.05
        summary['bert_multilingual_framing_eta_squared'] = bert_multilingual_results['framing_eta_squared']
        summary['bert_multilingual_category_eta_squared'] = bert_multilingual_results['category_eta_squared']
        
        return summary 