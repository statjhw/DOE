"""
RCBD 감정 프레이밍 실험 실행 스크립트

사용법:
    python run_experiment.py --mode full       # 전체 실험 실행
    python run_experiment.py --mode test       # 빠른 테스트 실행
    python run_experiment.py --mode demo       # 데모 실행 (매우 적은 데이터)
"""

import argparse
import sys
from datetime import datetime

from config import (
    OPENAI_API_KEY, DEFAULT_EXPERIMENT_PARAMS, 
    OUTPUT_DIR, EXPERIMENT_INFO, validate_config
)
from main_experiment import RCBDExperiment

def print_experiment_info():
    """실험 정보 출력"""
    info = EXPERIMENT_INFO
    print("\n" + "="*80)
    print(f"🔬 {info['title']}")
    print("="*80)
    print(f"📋 목표: {info['objective']}")
    print(f"📊 설계: {info['design']}")
    
    print(f"\n📈 실험 요인:")
    print(f"   • 처리 요인: {info['factors']['treatment']['name']}")
    print(f"     수준: {', '.join(info['factors']['treatment']['levels'])}")
    print(f"     설명: {info['factors']['treatment']['description']}")
    
    print(f"   • 블록 요인: {info['factors']['block']['name']}")
    print(f"     수준: {', '.join(info['factors']['block']['levels'])}")
    print(f"     설명: {info['factors']['block']['description']}")
    
    print(f"\n📊 반응변수:")
    print(f"   • 주 변수: {info['response_variables']['primary']}")
    print(f"   • 보조 변수: {info['response_variables']['secondary']}")
    print(f"   • 설명: {info['response_variables']['description']}")

def run_full_experiment(api_key):
    """전체 실험 실행"""
    print("\n🔬 전체 RCBD 실험 시작")
    print("📊 실험 규모: 351개 데이터 포인트 (3 카테고리 × 39 질문 × 3 프레이밍)")
    print("📝 예상 소요시간: 2-4시간")
    print("💸 예상 API 비용: $0.5-1 (GPT-4o-mini 기준)")
    print("   - 프롬프트 생성: 351개 (~$0.06)")
    print("   - AI 응답 수집: 1,404개 (~$0.35)")
    
    confirm = input("\n계속 진행하시겠습니까? (y/N): ")
    if confirm.lower() != 'y':
        print("실험이 취소되었습니다.")
        return
    
    # 실험 실행
    experiment = RCBDExperiment(api_key, output_dir=OUTPUT_DIR)
    results = experiment.run_complete_experiment(
        n_responses=5,
        temperature=0.7
    )
    
    print("\n✅ 전체 실험 완료!")
    return results

def run_test_experiment(api_key):
    """테스트 실험 실행 (각 카테고리당 1개 질문)"""
    print("\n🧪 테스트 실험 시작")
    print("📝 예상 소요시간: 5-10분")
    print("💸 예상 API 비용: $1-3")
    
    experiment = RCBDExperiment(api_key, output_dir=f"{OUTPUT_DIR}/test")
    results = experiment.run_quick_test(n_responses=3, n_questions_per_category=3)
    
    print("\n✅ 테스트 실험 완료!")
    return results

def run_demo_experiment(api_key):
    """데모 실험 실행 (최소한의 데이터)"""
    print("\n🎭 데모 실험 시작")
    print("📝 예상 소요시간: 2-3분")
    print("💸 예상 API 비용: $0.5-1")
    
    experiment = RCBDExperiment(api_key, output_dir=f"{OUTPUT_DIR}/demo")
    results = experiment.run_quick_test(n_responses=2, n_questions_per_category=1)
    
    print("\n✅ 데모 실험 완료!")
    return results

def main():
    parser = argparse.ArgumentParser(description='RCBD 감정 프레이밍 실험 실행')
    parser.add_argument(
        '--mode', 
        choices=['full', 'test', 'demo'], 
        default='test',
        help='실험 모드 선택 (default: test)'
    )
    
    args = parser.parse_args()
    
    # 실험 정보 출력
    print_experiment_info()
    
    # API 키 확인
    api_key = OPENAI_API_KEY
    
    # 실험 모드별 실행
    start_time = datetime.now()
    
    try:
        if args.mode == 'full':
            results = run_full_experiment(api_key)
        elif args.mode == 'test':
            results = run_test_experiment(api_key)
        elif args.mode == 'demo':
            results = run_demo_experiment(api_key)
        else:
            print(f"❌ 알 수 없는 모드: {args.mode}")
            sys.exit(1)
            
        # 실행 결과 요약
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n📊 실험 실행 요약")
        print(f"   • 모드: {args.mode}")
        print(f"   • 시작 시간: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • 종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • 소요 시간: {duration}")
        
        if results and 'rcbd_report' in results:
            df = results['rcbd_report']['dataframe']
            print(f"   • 전체 데이터 포인트: {len(df)}")
            print(f"   • 평균 BERT Multilingual 유사도: {df['bert_multilingual_similarity'].mean():.4f}")
            
            # 주요 결과 확인
            summary = results['rcbd_report']['summary']
            framing_significant = summary.get('bert_multilingual_framing_significant', False)
            category_significant = summary.get('bert_multilingual_category_significant', False)
            
            print(f"   • 프레이밍 효과: {'유의함' if framing_significant else '비유의함'}")
            print(f"   • 카테고리 효과: {'유의함' if category_significant else '비유의함'}")
        
        print(f"\n🎉 실험이 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 실험이 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 실험 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 