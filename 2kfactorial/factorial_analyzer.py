"""
2^4 Factorial Design 통계 분석기 (context_provision 제거)
카테고리를 블록 또는 요인으로 처리하는 유연한 분석
주효과와 교호작용 분석
ANOVA 결과: context_provision은 p=0.982로 효과 없음
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

class FactorialAnalyzer:
    def __init__(self):
        """2^4 Factorial 분석기 초기화"""
        pass
    
    def prepare_dataframe(self, experimental_data):
        """
        실험 데이터를 분석용 DataFrame으로 변환
        
        Args:
            experimental_data (list): 일관성 점수가 포함된 실험 데이터
            
        Returns:
            pd.DataFrame: 분석용 DataFrame
        """
        rows = []
        
        for data in experimental_data:
            if data.get('bert_multilingual_similarity') is not None:
                row = {
                    'condition_id': data['condition_id'],
                    'category': data['category'],
                    'base_question': data['base_question'],
                    'bert_multilingual_similarity': data['bert_multilingual_similarity'],
                    'n_responses': data.get('n_responses', 0)
                }
                
                # 각 요인을 개별 컬럼으로 추가
                for factor_name, factor_value in data['factor_combination'].items():
                    row[factor_name] = factor_value
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def analyze_with_category_as_block(self, df, response_var='bert_multilingual_similarity'):
        """
        카테고리를 블록으로 처리한 2^4 Factorial 분석
        
        Args:
            df (pd.DataFrame): 분석용 데이터프레임
            response_var (str): 반응변수
            
        Returns:
            dict: 분석 결과
        """
        print(f"\n{'='*80}")
        print(f"2^4 Factorial Analysis (Category as Block) - 반응변수: {response_var}")
        print(f"{'='*80}")
        
        print(f"전체 관측치 수: {len(df)}")
        print(f"블록(카테고리) 수: {df['category'].nunique()}")
        
        factors = ['prompt_language', 'model', 'role_assignment', 'explicitness']
        
        print(f"요인별 수준 수:")
        for factor in factors:
            if factor in df.columns:
                levels = df[factor].nunique()
                print(f"  • {factor}: {levels}개 수준")
        
        # 블록을 포함한 모델
        try:
            factor_terms = " * ".join([f"C({factor})" for factor in factors if factor in df.columns])
            formula = f"{response_var} ~ C(category) + {factor_terms}"
            print(f"\n모델 공식: {formula}")
            
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            print(f"\n📊 2^4 Factorial with Blocking ANOVA 결과 ({response_var})")
            print("-" * 60)
            print(anova_table)
            
            # 주효과 분석
            main_effects = self._analyze_main_effects(df, factors, response_var)
            
            # 블록 효과 분석
            block_effects = self._analyze_block_effects(df, response_var)
            
            results = {
                'model': model,
                'anova_table': anova_table,
                'main_effects': main_effects,
                'block_effects': block_effects,
                'response_variable': response_var,
                'analysis_type': 'category_as_block'
            }
            
        except Exception as e:
            print(f"\n⚠️  전체 모델 분석 실패: {e}")
            print("   주효과와 블록 효과만 분석합니다.")
            
            main_effects = self._analyze_main_effects(df, factors, response_var)
            block_effects = self._analyze_block_effects(df, response_var)
            
            results = {
                'model': None,
                'anova_table': None,
                'main_effects': main_effects,
                'block_effects': block_effects,
                'response_variable': response_var,
                'analysis_type': 'category_as_block',
                'error': str(e)
            }
        
        return results
    
    def analyze_with_category_as_factor(self, df, response_var='bert_multilingual_similarity'):
        """
        카테고리를 요인으로 처리한 2^4+1 Factorial 분석
        
        Args:
            df (pd.DataFrame): 분석용 데이터프레임
            response_var (str): 반응변수
            
        Returns:
            dict: 분석 결과
        """
        print(f"\n{'='*80}")
        print(f"2^4+1 Factorial Analysis (Category as Factor) - 반응변수: {response_var}")
        print(f"{'='*80}")
        
        print(f"전체 관측치 수: {len(df)}")
        
        factors = ['prompt_language', 'model', 'role_assignment', 'explicitness', 'category']
        
        print(f"요인별 수준 수:")
        for factor in factors:
            if factor in df.columns:
                levels = df[factor].nunique()
                print(f"  • {factor}: {levels}개 수준")
        
        # 카테고리를 요인으로 포함한 모델
        try:
            factor_terms = " * ".join([f"C({factor})" for factor in factors if factor in df.columns])
            formula = f"{response_var} ~ {factor_terms}"
            print(f"\n모델 공식: {formula}")
            
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            print(f"\n📊 2^4+1 Factorial ANOVA 결과 ({response_var})")
            print("-" * 60)
            print(anova_table)
            
            # 주효과 분석 (카테고리 포함)
            main_effects = self._analyze_main_effects(df, factors, response_var)
            
            # 2차 교호작용 분석
            interaction_effects = self._analyze_two_way_interactions(df, factors, response_var)
            
            results = {
                'model': model,
                'anova_table': anova_table,
                'main_effects': main_effects,
                'interaction_effects': interaction_effects,
                'response_variable': response_var,
                'analysis_type': 'category_as_factor'
            }
            
        except Exception as e:
            print(f"\n⚠️  전체 모델 분석 실패: {e}")
            print("   주효과만 분석합니다.")
            
            # 주효과만 분석
            main_effects = self._analyze_main_effects(df, factors, response_var)
            
            results = {
                'model': None,
                'anova_table': None,
                'main_effects': main_effects,
                'interaction_effects': None,
                'response_variable': response_var,
                'analysis_type': 'category_as_factor',
                'error': str(e)
            }
        
        return results
    
    def _analyze_main_effects(self, df, factors, response_var):
        """주효과 분석"""
        main_effects = {}
        
        print(f"\n📈 주효과 분석")
        print("-" * 40)
        
        for factor in factors:
            if factor in df.columns:
                factor_means = df.groupby(factor)[response_var].agg(['mean', 'std', 'count']).round(4)
                main_effects[factor] = factor_means
                
                effect_size = factor_means['mean'].max() - factor_means['mean'].min()
                
                print(f"\n🔹 {factor}")
                print(factor_means)
                print(f"   효과 크기: {effect_size:.4f}")
        
        return main_effects
    
    def _analyze_two_way_interactions(self, df, factors, response_var):
        """2차 교호작용 분석"""
        interactions = {}
        
        print(f"\n🔄 2차 교호작용 분석")
        print("-" * 40)
        
        for factor1, factor2 in combinations(factors, 2):
            if factor1 in df.columns and factor2 in df.columns:
                interaction_means = df.groupby([factor1, factor2])[response_var].mean().unstack()
                interactions[f"{factor1}_x_{factor2}"] = interaction_means
                
                print(f"\n🔸 {factor1} × {factor2}")
                print(interaction_means.round(4))
        
        return interactions
    
    def _analyze_block_effects(self, df, response_var):
        """블록 효과 분석"""
        print(f"\n🧱 블록(카테고리) 효과 분석")
        print("-" * 40)
        
        block_means = df.groupby('category')[response_var].agg(['mean', 'std', 'count']).round(4)
        
        print(block_means)
        
        block_effect_size = block_means['mean'].max() - block_means['mean'].min()
        print(f"\n블록 효과 크기: {block_effect_size:.4f}")
        
        return {
            'block_means': block_means,
            'block_effect_size': block_effect_size
        }
    
    def compare_analysis_approaches(self, df, response_var='bert_multilingual_similarity'):
        """
        블록 분석 vs 요인 분석 비교
        
        Args:
            df (pd.DataFrame): 분석용 데이터프레임
            response_var (str): 반응변수
            
        Returns:
            dict: 비교 결과
        """
        print(f"\n{'='*80}")
        print(f"분석 방법 비교: 블록 vs 요인")
        print(f"{'='*80}")
        
        # 블록 분석
        block_results = self.analyze_with_category_as_block(df, response_var)
        
        # 요인 분석
        factor_results = self.analyze_with_category_as_factor(df, response_var)
        
        # 비교 요약
        comparison = {
            'block_analysis': block_results,
            'factor_analysis': factor_results,
            'comparison_summary': {
                'block_r_squared': block_results['model'].rsquared if block_results['model'] else None,
                'factor_r_squared': factor_results['model'].rsquared if factor_results['model'] else None,
                'block_aic': block_results['model'].aic if block_results['model'] else None,
                'factor_aic': factor_results['model'].aic if factor_results['model'] else None
            }
        }
        
        print(f"\n📊 분석 방법 비교 요약")
        print("-" * 40)
        if comparison['comparison_summary']['block_r_squared']:
            print(f"블록 분석 R²: {comparison['comparison_summary']['block_r_squared']:.4f}")
        if comparison['comparison_summary']['factor_r_squared']:
            print(f"요인 분석 R²: {comparison['comparison_summary']['factor_r_squared']:.4f}")
        
        return comparison
    
    def generate_factorial_report(self, experimental_data):
        """
        종합 분석 보고서 생성
        
        Args:
            experimental_data (list): 실험 데이터
            
        Returns:
            dict: 종합 분석 결과
        """
        print(f"\n🔬 2^4 Factorial Design 종합 분석 보고서 생성")
        print("=" * 80)
        
        # DataFrame 생성
        df = self.prepare_dataframe(experimental_data)
        
        if len(df) == 0:
            print("⚠️  분석할 데이터가 없습니다.")
            return {
                'dataframe': df,
                'analysis_results': None,
                'error': 'No data to analyze'
            }
        
        print(f"분석 대상 데이터: {len(df)}개 관측치")
        
        # 두 가지 분석 방법 모두 수행
        analysis_results = self.compare_analysis_approaches(df)
        
        # 설계 요약
        design_summary = {
            'total_conditions': len(df),
            'factors': ['prompt_language', 'model', 'role_assignment', 'explicitness'],
            'categories': df['category'].unique().tolist(),
            'factor_combinations': len(df.groupby(['prompt_language', 'model', 'role_assignment', 'explicitness'])),
            'questions_per_category': df.groupby('category').size().to_dict()
        }
        
        return {
            'dataframe': df,
            'analysis_results': analysis_results,
            'design_summary': design_summary,
            'main_effects': analysis_results['block_analysis']['main_effects'] if analysis_results['block_analysis']['main_effects'] else None
        }

if __name__ == "__main__":
    print("🧪 2^4 Factorial Design 분석기 테스트")
    print("=" * 60)
    
    # 테스트용 샘플 데이터 생성
    sample_data = [
        {
            'condition_id': 1,
            'category': '인성',
            'base_question': '테스트 질문',
            'factor_combination': {
                'prompt_language': 'korean',
                'model': 'gpt-4o-mini',
                'role_assignment': 'with_role',
                'explicitness': 'low'
            },
            'bert_multilingual_similarity': 0.85,
            'n_responses': 3
        }
    ]
    
    analyzer = FactorialAnalyzer()
    df = analyzer.prepare_dataframe(sample_data)
    
    print(f"샘플 데이터프레임 생성 완료: {len(df)}개 행")
    print(f"컬럼: {list(df.columns)}")
    
    print("\n✅ 분석기 초기화 완료!") 