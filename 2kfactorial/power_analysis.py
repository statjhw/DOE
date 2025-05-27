"""
2^4 Factorial Design Power Analysis (context_provision 제거)
Representative 모드 실험 결과를 바탕으로 적정 샘플 크기 결정
ANOVA 결과: context_provision은 p=0.982로 효과 없음
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, ttest_power
import json
import ast
import glob
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = ['Arial Unicode MS', 'AppleGothic', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False

def load_representative_results():
    """Representative 모드 실험 결과 로드"""
    print("📊 Representative 모드 실험 결과 로드 중...")
    
    # 실험 데이터 로드
    df = pd.read_csv('factorial_results/factorial_experiment_data_representative_20250525_011508.csv')
    
    # factor_combination 문자열을 파싱해서 개별 컬럼으로 분해
    print("🔧 요인 조합 파싱 중...")
    
    # factor_combination을 JSON으로 파싱 (문자열이 dict 형태로 저장됨)
    factor_columns = []
    for idx, row in df.iterrows():
        try:
            # 문자열을 딕셔너리로 변환
            factor_dict = ast.literal_eval(row['factor_combination'])
            factor_columns.append(factor_dict)
        except:
            # 실패하면 빈 딕셔너리
            factor_columns.append({})
    
    # 요인별로 컬럼 생성
    factor_df = pd.DataFrame(factor_columns)
    
    # 원본 데이터프레임과 병합
    for col in factor_df.columns:
        df[col] = factor_df[col]
    
    print(f"✅ 요인 컬럼 추가 완료: {list(factor_df.columns)}")
    
    # 분석 데이터 로드
    analysis_df = pd.read_csv('factorial_results/factorial_analysis_data_representative_20250525_011508.csv')
    
    # 요약 통계 로드
    with open('factorial_results/factorial_experiment_summary_representative_20250525_011508.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    print(f"✅ 데이터 로드 완료")
    print(f"   - 실험 데이터: {len(df)}개 관측치")
    print(f"   - 분석 데이터: {len(analysis_df)}개 관측치")
    print(f"   - 요인 컬럼: {[col for col in df.columns if col in ['prompt_language', 'model', 'role_assignment', 'explicitness']]}")
    
    return df, analysis_df, summary

def load_demo_results():
    """Demo 모드 실험 결과 로드 (최신 파일 자동 검색)"""
    print("📊 Demo 모드 실험 결과 로드 중...")
    
    # 최신 demo 실험 파일들 찾기
    experiment_files = glob.glob('factorial_results/factorial_experiment_data_demo_*.csv')
    analysis_files = glob.glob('factorial_results/factorial_analysis_data_demo_*.csv')
    summary_files = glob.glob('factorial_results/factorial_experiment_summary_demo_*.json')
    
    if not experiment_files:
        raise FileNotFoundError("Demo 실험 데이터 파일을 찾을 수 없습니다.")
    
    # 가장 최근 파일 선택
    latest_experiment = max(experiment_files, key=os.path.getmtime)
    latest_analysis = max(analysis_files, key=os.path.getmtime) if analysis_files else None
    latest_summary = max(summary_files, key=os.path.getmtime) if summary_files else None
    
    print(f"📁 로드할 파일들:")
    print(f"   - 실험 데이터: {latest_experiment}")
    print(f"   - 분석 데이터: {latest_analysis}")
    print(f"   - 요약 통계: {latest_summary}")
    
    # 실험 데이터 로드
    df = pd.read_csv(latest_experiment)
    
    # factor_combination 문자열을 파싱해서 개별 컬럼으로 분해
    print("🔧 요인 조합 파싱 중...")
    
    # factor_combination을 JSON으로 파싱 (문자열이 dict 형태로 저장됨)
    factor_columns = []
    for idx, row in df.iterrows():
        try:
            # 문자열을 딕셔너리로 변환
            factor_dict = ast.literal_eval(row['factor_combination'])
            factor_columns.append(factor_dict)
        except:
            # 실패하면 빈 딕셔너리
            factor_columns.append({})
    
    # 요인별로 컬럼 생성
    factor_df = pd.DataFrame(factor_columns)
    
    # 원본 데이터프레임과 병합
    for col in factor_df.columns:
        df[col] = factor_df[col]
    
    print(f"✅ 요인 컬럼 추가 완료: {list(factor_df.columns)}")
    
    # 분석 데이터 로드 (있는 경우)
    analysis_df = None
    if latest_analysis:
        analysis_df = pd.read_csv(latest_analysis)
    
    # 요약 통계 로드 (있는 경우)
    summary = None
    if latest_summary:
        with open(latest_summary, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    
    print(f"✅ 데이터 로드 완료")
    print(f"   - 실험 데이터: {len(df)}개 관측치")
    if analysis_df is not None:
        print(f"   - 분석 데이터: {len(analysis_df)}개 관측치")
    print(f"   - 요인 컬럼: {[col for col in df.columns if col in ['prompt_language', 'model', 'role_assignment', 'explicitness']]}")
    
    return df, analysis_df, summary

def calculate_effect_sizes(df):
    """요인별 효과 크기 계산 (Cohen's d)"""
    print("\n📈 요인별 효과 크기 계산")
    print("=" * 50)
    
    effect_sizes = {}
    
    # 각 요인별로 Cohen's d 계산
    factors = ['prompt_language', 'model', 'role_assignment', 'explicitness']
    
    for factor in factors:
        groups = df.groupby(factor)['bert_multilingual_similarity']
        
        if len(groups) == 2:  # 2수준 요인
            group_names = list(groups.groups.keys())
            group1 = groups.get_group(group_names[0])
            group2 = groups.get_group(group_names[1])
            
            # Cohen's d 계산
            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                (len(group2) - 1) * group2.var()) / 
                               (len(group1) + len(group2) - 2))
            
            cohens_d = abs(group1.mean() - group2.mean()) / pooled_std
            
            effect_sizes[factor] = {
                'cohens_d': cohens_d,
                'mean_diff': abs(group1.mean() - group2.mean()),
                'group1_mean': group1.mean(),
                'group2_mean': group2.mean(),
                'pooled_std': pooled_std,
                'interpretation': interpret_effect_size(cohens_d)
            }
            
            print(f"🎯 {factor}:")
            print(f"   Cohen's d: {cohens_d:.4f} ({effect_sizes[factor]['interpretation']})")
            print(f"   평균 차이: {effect_sizes[factor]['mean_diff']:.4f}")
            print(f"   그룹 평균: {group1.mean():.4f} vs {group2.mean():.4f}")
    
    # 카테고리 효과 (3수준)
    category_groups = df.groupby('category')['bert_multilingual_similarity']
    category_means = [group.mean() for _, group in category_groups]
    category_effect = (max(category_means) - min(category_means)) / df['bert_multilingual_similarity'].std()
    
    effect_sizes['category'] = {
        'effect_size': category_effect,
        'interpretation': interpret_effect_size(category_effect),
        'means': {name: group.mean() for name, group in category_groups}
    }
    
    print(f"🎯 category:")
    print(f"   Effect size: {category_effect:.4f} ({effect_sizes['category']['interpretation']})")
    print(f"   카테고리별 평균: {effect_sizes['category']['means']}")
    
    return effect_sizes

def interpret_effect_size(d):
    """Effect size 해석"""
    if d < 0.2:
        return "매우 작음"
    elif d < 0.5:
        return "작음"
    elif d < 0.8:
        return "중간"
    else:
        return "큰 효과"

def power_analysis_for_factors(effect_sizes, alpha=0.05):
    """요인별 power analysis"""
    print(f"\n⚡ Power Analysis (α = {alpha})")
    print("=" * 50)
    
    power_results = {}
    
    # 목표 power 수준들
    target_powers = [0.8, 0.9, 0.95]
    
    for factor, effect_data in effect_sizes.items():
        if factor == 'category':
            continue  # 카테고리는 별도 처리
            
        cohens_d = effect_data['cohens_d']
        
        print(f"\n🎯 {factor} (Cohen's d = {cohens_d:.4f})")
        
        factor_results = {}
        
        for power in target_powers:
            # t-test에 필요한 샘플 크기 계산 (간단한 공식 사용)
            # Cohen's power tables 기반 근사식
            if power == 0.8:
                if cohens_d < 0.2:
                    n_per_group = 400  # 매우 작은 효과
                elif cohens_d < 0.5:
                    n_per_group = 64   # 작은 효과
                elif cohens_d < 0.8:
                    n_per_group = 26   # 중간 효과  
                else:
                    n_per_group = 17   # 큰 효과
            elif power == 0.9:
                if cohens_d < 0.2:
                    n_per_group = 526
                elif cohens_d < 0.5:
                    n_per_group = 85
                elif cohens_d < 0.8:
                    n_per_group = 34
                else:
                    n_per_group = 22
            else:  # power == 0.95
                if cohens_d < 0.2:
                    n_per_group = 651
                elif cohens_d < 0.5:
                    n_per_group = 105
                elif cohens_d < 0.8:
                    n_per_group = 42
                else:
                    n_per_group = 27
            
            # 더 정확한 계산 (scipy 사용)
            try:
                from scipy.stats import norm
                z_alpha = norm.ppf(1 - alpha/2)  # 양측검정
                z_beta = norm.ppf(power)
                n_per_group = ((z_alpha + z_beta) / cohens_d) ** 2
            except:
                pass  # 위의 근사값 사용
            
            total_n = n_per_group * 2
            
            # 2^4 factorial에서 필요한 총 조건 수 계산
            # 16개 조합 중 절반씩이 각 수준에 해당
            total_conditions = total_n * 8  # 16/2 = 8개 조합씩
            
            # 카테고리 고려 (3개)
            total_with_categories = total_conditions * 3
            
            factor_results[power] = {
                'n_per_group': n_per_group,
                'total_n': total_n,
                'total_conditions': total_conditions,
                'total_with_categories': total_with_categories
            }
            
            print(f"   Power {power}: 그룹당 {n_per_group:.1f}개, 총 {total_with_categories:.0f}개 조건")
        
        power_results[factor] = factor_results
    
    return power_results

def cost_analysis(power_results, cost_per_condition=0.01):
    """비용 분석"""
    print(f"\n💰 비용 분석 (조건당 ${cost_per_condition})")
    print("=" * 50)
    
    cost_analysis_results = {}
    
    for factor, power_data in power_results.items():
        print(f"\n🎯 {factor}:")
        
        factor_costs = {}
        for power, sample_data in power_data.items():
            total_cost = sample_data['total_with_categories'] * cost_per_condition
            factor_costs[power] = total_cost
            
            print(f"   Power {power}: ${total_cost:.2f} (조건 {sample_data['total_with_categories']:.0f}개)")
        
        cost_analysis_results[factor] = factor_costs
    
    return cost_analysis_results

def current_power_analysis(effect_sizes, current_n=96):
    """현재 샘플 크기(96)에서의 power 계산"""
    print(f"\n🔍 현재 설계의 Power 분석 (n={current_n})")
    print("=" * 50)
    
    current_powers = {}
    
    for factor, effect_data in effect_sizes.items():
        if factor == 'category':
            continue
            
        cohens_d = effect_data['cohens_d']
        
        # 현재 그룹당 샘플 크기 (96개 조건의 절반씩)
        n_per_group = current_n / 2
        
        # 현재 power 계산 - 올바른 파라미터만 사용
        current_power = ttest_power(effect_size=cohens_d, nobs=n_per_group, alpha=0.05, alternative='two-sided')
        
        current_powers[factor] = current_power
        
        print(f"🎯 {factor}:")
        print(f"   Effect size (Cohen's d): {cohens_d:.4f}")
        print(f"   현재 Power: {current_power:.3f}")
        print(f"   Power 충분성: {'✅ 충분' if current_power >= 0.8 else '❌ 부족'}")
    
    return current_powers

def plot_power_curves(effect_sizes):
    """Power curve 시각화"""
    print(f"\n📊 Power Curve 시각화")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    sample_sizes = np.arange(10, 200, 5)
    
    factor_idx = 0
    for factor, effect_data in effect_sizes.items():
        if factor == 'category' or factor_idx >= 6:
            continue
            
        cohens_d = effect_data['cohens_d']
        
        # 각 샘플 크기에 대한 power 계산
        powers = []
        for n in sample_sizes:
            power = ttest_power(effect_size=cohens_d, nobs=n/2, alpha=0.05, alternative='two-sided')
            powers.append(power)
        
        ax = axes[factor_idx]
        ax.plot(sample_sizes, powers, linewidth=2, label=f'{factor}')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power 0.8')
        ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='Power 0.9')
        ax.axvline(x=96, color='green', linestyle='-', alpha=0.7, label='현재 설계 (96)')
        
        ax.set_xlabel('총 샘플 크기')
        ax.set_ylabel('Statistical Power')
        ax.set_title(f'{factor}\n(Cohen\'s d = {cohens_d:.3f})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)
        
        factor_idx += 1
    
    # 빈 subplot 제거
    for i in range(factor_idx, 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('factorial_results/power_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_recommendations(effect_sizes, power_results, current_powers):
    """실험 설계 추천안 생성"""
    print(f"\n🎯 실험 설계 추천안")
    print("=" * 60)
    
    # 효과 크기별 요인 분류
    large_effects = []
    medium_effects = []
    small_effects = []
    
    for factor, effect_data in effect_sizes.items():
        if factor == 'category':
            continue
        
        cohens_d = effect_data['cohens_d']
        if cohens_d >= 0.8:
            large_effects.append(factor)
        elif cohens_d >= 0.5:
            medium_effects.append(factor)
        elif cohens_d >= 0.2:
            small_effects.append(factor)
    
    print(f"📊 효과 크기별 요인 분류:")
    print(f"   큰 효과 (d≥0.8): {large_effects}")
    print(f"   중간 효과 (0.5≤d<0.8): {medium_effects}")  
    print(f"   작은 효과 (0.2≤d<0.5): {small_effects}")
    
    # 현재 power가 부족한 요인들
    insufficient_power = [factor for factor, power in current_powers.items() if power < 0.8]
    
    print(f"\n⚡ Power 분석 결과:")
    print(f"   Power 부족 요인: {insufficient_power}")
    print(f"   Power 충분 요인: {[f for f in current_powers.keys() if f not in insufficient_power]}")
    
    # 추천안
    print(f"\n💡 추천안:")
    
    if not insufficient_power:
        print("   ✅ 현재 Representative 설계(96개 조건)로 충분합니다!")
        print("   ✅ 모든 주요 요인에 대해 적절한 statistical power를 확보했습니다.")
        recommendation = "representative"
    elif len(insufficient_power) <= 2:
        print("   📈 Test 설계를 Representative 설계로 확장하는 것을 추천합니다.")
        print("   📈 현재 설계가 대부분 요인에 충분한 power를 제공합니다.")
        recommendation = "representative"
    else:
        print("   📈 Full 설계(480개 조건)를 고려해보세요.")
        print("   📈 더 많은 요인에 대해 충분한 statistical power가 필요합니다.")
        recommendation = "full"
    
    # 비용 효율성
    print(f"\n💰 비용 효율성:")
    print(f"   Representative (96조건): ~$0.40")
    print(f"   Full (480조건): ~$2.00")
    print(f"   추천: {recommendation} 설계")
    
    return {
        'large_effects': large_effects,
        'medium_effects': medium_effects,
        'small_effects': small_effects,
        'insufficient_power': insufficient_power,
        'recommendation': recommendation
    }

def main():
    """메인 분석 함수"""
    print("🔬 2^4 Factorial Design Power Analysis")
    print("=" * 60)
    
    # 1. 데이터 로드
    df, analysis_df, summary = load_demo_results()
    
    # 2. 효과 크기 계산
    effect_sizes = calculate_effect_sizes(df)
    
    # 3. Power analysis
    power_results = power_analysis_for_factors(effect_sizes)
    
    # 4. 비용 분석
    cost_results = cost_analysis(power_results)
    
    # 5. 현재 설계의 power 분석
    current_powers = current_power_analysis(effect_sizes)
    
    # 6. Power curve 시각화
    plot_power_curves(effect_sizes)
    
    # 7. 추천안 생성
    recommendations = generate_recommendations(effect_sizes, power_results, current_powers)
    
    # 8. 결과 저장
    results = {
        'effect_sizes': effect_sizes,
        'power_results': power_results,
        'cost_results': cost_results,
        'current_powers': current_powers,
        'recommendations': recommendations
    }
    
    # JSON으로 저장 (numpy 타입 처리)
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    with open('factorial_results/power_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=convert_numpy)
    
    print(f"\n💾 결과 저장 완료:")
    print(f"   📊 Power curves: factorial_results/power_curves.png")
    print(f"   📄 분석 결과: factorial_results/power_analysis_results.json")
    
    return results

if __name__ == "__main__":
    results = main() 