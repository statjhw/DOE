"""
2^4 Factorial Design 메인 실험 실행기 (context_provision 제거)
ANOVA 결과: context_provision은 p=0.982로 효과 없음
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

from prompt_generator import FactorialPromptGenerator
from response_collector import FactorialResponseCollector
from consistency_analyzer import ConsistencyAnalyzer
from factorial_analyzer import FactorialAnalyzer
from config import get_experiment_config, EXPERIMENT_MODES
from experiment_design import get_design_summary

class FactorialExperimentRunner:
    def __init__(self, api_key: str, output_dir: str = "factorial_results"):
        """
        2^4 Factorial 실험 실행기 초기화
        
        Args:
            api_key (str): OpenAI API 키
            output_dir (str): 결과 저장 디렉토리
        """
        self.api_key = api_key
        self.output_dir = output_dir
        
        # 컴포넌트 초기화
        self.prompt_generator = FactorialPromptGenerator(api_key)
        self.response_collector = FactorialResponseCollector(api_key)
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.factorial_analyzer = FactorialAnalyzer()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def run_experiment(self, mode: str = 'demo', save_intermediate: bool = True) -> Dict:
        """
        전체 실험 실행
        
        Args:
            mode (str): 실험 모드 ('demo', 'test', 'representative', 'full')
            save_intermediate (bool): 중간 결과 저장 여부
            
        Returns:
            Dict: 실험 결과
        """
        print(f"\n🚀 2^4 Factorial Design 실험 시작")
        print("=" * 80)
        
        # 설정 로드
        config = get_experiment_config(mode)
        
        print(f"실험 모드: {config['mode_name']}")
        print(f"테스트 요인: {len(config['factors_to_test'])}개")
        print(f"최대 조건 수: {config['max_conditions']}")
        
        # 1단계: 프롬프트 생성
        print(f"\n📝 1단계: 프롬프트 생성")
        print("-" * 50)
        
        if mode == 'full':
            experimental_data = self.prompt_generator.generate_full_prompts()
        elif mode == 'representative':
            experimental_data = self.prompt_generator.generate_representative_prompts()
        else:  # demo, test
            experimental_data = self.prompt_generator.generate_subset_prompts(
                config['factors_to_test'], 
                config['questions_per_category']
            )
        
        # 조건 수 제한 (필요시)
        if len(experimental_data) > config['max_conditions']:
            experimental_data = experimental_data[:config['max_conditions']]
        
        print(f"생성된 조건 수: {len(experimental_data)}")
        
        if save_intermediate:
            self._save_intermediate_data(experimental_data, f"01_{mode}_prompts_generated.json")
        
        # 2단계: 응답 수집
        print(f"\n💬 2단계: AI 응답 수집")
        print("-" * 50)
        
        experimental_data = self.response_collector.collect_responses_for_experiment(
            experimental_data, 
            n_responses=config['n_responses'],
            temperature=config['temperature']
        )
        
        if save_intermediate:
            self._save_intermediate_data(experimental_data, f"02_{mode}_responses_collected.json")
        
        # 3단계: 일관성 분석
        print(f"\n🔍 3단계: 일관성 분석")
        print("-" * 50)
        
        experimental_data = self.consistency_analyzer.analyze_consistency_for_experiment(experimental_data)
        
        if save_intermediate:
            self._save_intermediate_data(experimental_data, f"03_{mode}_consistency_analyzed.json")
        
        # 4단계: 통계 분석
        print(f"\n📊 4단계: 2^4 Factorial 통계 분석")
        print("-" * 50)
        
        factorial_report = self.factorial_analyzer.generate_factorial_report(experimental_data)
        
        # 일관성 통계 계산
        consistency_stats = self._calculate_consistency_statistics(experimental_data)
        
        # 5단계: 결과 저장
        print(f"\n💾 5단계: 결과 저장")
        print("-" * 50)
        
        results = {
            'experimental_data': experimental_data,
            'factorial_report': factorial_report,
            'consistency_stats': consistency_stats,
            'config': config
        }
        
        self._save_final_results(results, mode)
        
        # 요약 출력
        self._print_experiment_summary(results)
        
        return results
    
    def _calculate_consistency_statistics(self, experimental_data: List[Dict]) -> Dict:
        """일관성 점수 통계 계산"""
        similarities = [
            data['bert_multilingual_similarity'] 
            for data in experimental_data 
            if data.get('bert_multilingual_similarity') is not None
        ]
        
        if not similarities:
            return {}
        
        return {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'count': len(similarities)
        }
    
    def _save_intermediate_data(self, data: List[Dict], filename: str):
        """중간 결과 저장"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        print(f"   중간 결과 저장: {filename}")
    
    def _save_final_results(self, results: Dict, mode: str):
        """최종 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 실험 데이터 CSV
        df = pd.DataFrame(results['experimental_data'])
        csv_path = os.path.join(self.output_dir, f"factorial_experiment_data_{mode}_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   실험 데이터 저장: factorial_experiment_data_{mode}_{timestamp}.csv")
        
        # 2. 분석 결과 DataFrame
        if 'dataframe' in results['factorial_report']:
            analysis_df = results['factorial_report']['dataframe']
            analysis_csv_path = os.path.join(self.output_dir, f"factorial_analysis_data_{mode}_{timestamp}.csv")
            analysis_df.to_csv(analysis_csv_path, index=False, encoding='utf-8-sig')
            print(f"   분석 데이터 저장: factorial_analysis_data_{mode}_{timestamp}.csv")
        
        # 3. 요약 통계 JSON
        def convert_to_serializable(obj):
            """JSON 직렬화 가능한 형태로 변환"""
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        summary_data = {
            'consistency_statistics': convert_to_serializable(results['consistency_stats']),
            'design_summary': convert_to_serializable(results['factorial_report'].get('design_summary', {})),
            'config': convert_to_serializable(results['config']),
            'timestamp': timestamp,
            'mode': mode
        }
        
        summary_path = os.path.join(self.output_dir, f"factorial_experiment_summary_{mode}_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)
        print(f"   요약 통계 저장: factorial_experiment_summary_{mode}_{timestamp}.json")
    
    def _print_experiment_summary(self, results: Dict):
        """실험 요약 출력"""
        print(f"\n🎉 2^4 Factorial Design 실험 완료!")
        print("=" * 80)
        
        # 기본 정보
        config = results['config']
        consistency_stats = results['consistency_stats']
        
        print(f"📊 실험 결과 요약:")
        print(f"   • 실험 모드: {config['mode_name']}")
        print(f"   • 총 조건 수: {len(results['experimental_data'])}")
        print(f"   • 총 응답 수: {len(results['experimental_data']) * config['n_responses']}")
        
        if consistency_stats:
            print(f"\n🔍 일관성 분석 결과:")
            print(f"   • 평균 BERT Multilingual 유사도: {consistency_stats['mean_similarity']:.4f}")
            print(f"   • 표준편차: {consistency_stats['std_similarity']:.4f}")
            print(f"   • 범위: {consistency_stats['min_similarity']:.4f} ~ {consistency_stats['max_similarity']:.4f}")
        
        # 요인별 주요 결과 (간단히)
        factorial_report = results['factorial_report']
        
        if 'main_effects' in factorial_report:
            print(f"\n📈 주요 요인 효과:")
            main_effects = factorial_report['main_effects']
            for factor, effect_data in main_effects.items():
                if isinstance(effect_data, dict) and 'effect_size' in effect_data:
                    print(f"   • {factor}: 효과 크기 {effect_data['effect_size']:.4f}")
        
        print(f"\n✅ 결과 파일들이 '{self.output_dir}' 폴더에 저장되었습니다.")

# 실행 예시
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("🧪 2^4 Factorial Design 실험 실행기 테스트")
    print("=" * 60)
    
    runner = FactorialExperimentRunner(api_key)
    
    # 데모 실험 실행
    print("\n데모 실험을 실행하시겠습니까? (y/n): ", end="")
    if input().lower() == 'y':
        results = runner.run_experiment(mode='demo')
        print(f"\n실험 완료! 결과는 {runner.output_dir} 폴더를 확인하세요.")
    else:
        print("실험을 건너뛰었습니다.") 