"""
2^4 Factorial Design 실험 실행 스크립트 (context_provision 제거)
ANOVA 결과: context_provision은 p=0.982로 효과 없음
"""

import os
import argparse
from dotenv import load_dotenv

from main_experiment import FactorialExperimentRunner
from config import print_experiment_info, get_experiment_config, estimate_experiment_cost

def run_factorial_experiment(mode: str = 'demo', save_intermediate: bool = True):
    """
    2^4 Factorial 실험 실행
    
    Args:
        mode (str): 실험 모드
        save_intermediate (bool): 중간 결과 저장 여부
    """
    # API 키 로드
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=your-api-key-here 를 추가하세요.")
        return None
    
    # 실험 정보 출력
    print_experiment_info(mode)
    
    # 비용 확인 및 사용자 동의
    cost_info = estimate_experiment_cost(mode)
    print(f"\n💰 예상 비용: ${cost_info['estimated_total_cost']:.3f}")
    print(f"📞 총 API 호출: {cost_info['total_requests']}회")
    
    if mode not in ['demo']:  # 데모가 아닌 경우 확인
        print(f"\n위 비용으로 실험을 진행하시겠습니까? (y/n): ", end="")
        if input().lower() != 'y':
            print("실험을 취소했습니다.")
            return None
    
    # 실험 실행
    runner = FactorialExperimentRunner(api_key)
    
    try:
        results = runner.run_experiment(mode=mode, save_intermediate=save_intermediate)
        
        # 결과 요약 출력
        print(f"\n🎯 실험 완료 요약:")
        print(f"   • 모드: {mode}")
        print(f"   • 총 조건 수: {len(results['experimental_data'])}")
        
        if results['consistency_stats']:
            print(f"   • 평균 BERT Multilingual 유사도: {results['consistency_stats']['mean_similarity']:.4f}")
        
        # 주요 요인 효과 (간단히)
        factorial_report = results['factorial_report']
        
        if 'main_effects' in factorial_report and factorial_report['main_effects']:
            print(f"\n📈 주요 요인 효과 크기:")
            for factor, factor_data in factorial_report['main_effects'].items():
                if hasattr(factor_data, 'mean'):
                    effect_range = factor_data['mean'].max() - factor_data['mean'].min()
                    print(f"   • {factor}: {effect_range:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 실험 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='2^4 Factorial Design AI 프롬프트 실험')
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'test', 'representative', 'full'],
        default='demo',
        help='실험 모드 선택'
    )
    
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='중간 결과 저장 안함'
    )
    
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='실험 정보만 출력하고 종료'
    )
    
    args = parser.parse_args()
    
    if args.info_only:
        # 모든 모드의 정보 출력
        print("🔬 2^4 Factorial Design 실험 모드별 정보")
        print("=" * 80)
        
        for mode in ['demo', 'test', 'representative', 'full']:
            print_experiment_info(mode)
            print()
        return
    
    # 실험 실행
    save_intermediate = not args.no_intermediate
    
    print(f"🚀 2^4 Factorial Design 실험 시작")
    print(f"   모드: {args.mode}")
    print(f"   중간 저장: {save_intermediate}")
    
    results = run_factorial_experiment(
        mode=args.mode,
        save_intermediate=save_intermediate
    )
    
    if results:
        print(f"\n✅ 실험이 성공적으로 완료되었습니다!")
        print(f"   결과 파일들을 'factorial_results' 폴더에서 확인하세요.")
    else:
        print(f"\n❌ 실험이 실패했습니다.")

if __name__ == "__main__":
    main() 