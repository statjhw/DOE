"""
RCBD 감정 프레이밍 실험 메인 실행 파일

실험 목표: 감정 프레이밍에 따른 AI 응답 일관성 변화 분석
- 독립변수: 프레이밍 수준 (중립적, 정서적, 자극적)
- 블록 요인: 질문 카테고리 (인성, 창의성, 논리적추론)
- 종속변수: 응답 일관성 (코사인 유사도, BERTScore)
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from config import OPENAI_API_KEY

from experiment_design import QUESTION_BLOCKS
from prompt_generator import FramingPromptGenerator
from response_collector import ResponseCollector
from consistency_analyzer import ConsistencyAnalyzer
from rcbd_analyzer import RCBDAnalyzer

class RCBDExperiment:
    def __init__(self, api_key, output_dir="results"):
        """
        RCBD 실험 매니저 초기화
        
        Args:
            api_key (str): OpenAI API 키
            output_dir (str): 결과 저장 디렉토리
        """
        self.api_key = api_key
        self.output_dir = output_dir
        
        # 모듈 초기화
        self.prompt_generator = FramingPromptGenerator(api_key)
        self.response_collector = ResponseCollector(api_key)
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.rcbd_analyzer = RCBDAnalyzer()
        
        # 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 실험 로그
        self.experiment_log = []
        
    def run_complete_experiment(self, n_responses=5, temperature=0.7, save_intermediate=True):
        """
                전체 RCBD 실험 실행

        Args:
            n_responses (int): 각 프롬프트당 수집할 응답 수
            temperature (float): AI 응답 다양성 파라미터
            save_intermediate (bool): 중간 결과 저장 여부
            
        Returns:
            dict: 종합 실험 결과
        """
        print("🔬 RCBD 감정 프레이밍 실험 시작")
        print("="*80)
        
        start_time = datetime.now()
        self._log(f"실험 시작: {start_time}")
        
        # 1단계: 프롬프트 생성
        print("\n📝 1단계: 프레이밍 프롬프트 생성")
        experimental_data = self.prompt_generator.batch_generate_framings(QUESTION_BLOCKS)
        
        if save_intermediate:
            self._save_json(experimental_data, "01_prompts_generated.json")
        
        self._log(f"프롬프트 생성 완료: {len(experimental_data)}개")
        
        # 2단계: 응답 수집
        print(f"\n💬 2단계: AI 응답 수집 (각 프롬프트당 {n_responses}개)")
        experimental_data = self.response_collector.collect_responses_by_condition(
            experimental_data, n_responses=n_responses, temperature=temperature
        )
        
        if save_intermediate:
            self._save_json(experimental_data, "02_responses_collected.json")
        
        self._log(f"응답 수집 완료: 총 {len(experimental_data) * n_responses}개 응답")
        
        # 3단계: 일관성 분석
        print("\n📊 3단계: 응답 일관성 분석")
        experimental_data = self.consistency_analyzer.analyze_consistency(experimental_data)
        
        if save_intermediate:
            self._save_json(experimental_data, "03_consistency_analyzed.json")
        
        # 일관성 통계 요약
        consistency_stats = self.consistency_analyzer.compute_consistency_statistics(experimental_data)
        self.consistency_analyzer.print_consistency_summary(consistency_stats)
        
        # 4단계: RCBD 통계 분석
        print("\n📈 4단계: RCBD 통계 분석")
        rcbd_report = self.rcbd_analyzer.generate_comprehensive_report(experimental_data)
        
        # 5단계: 결과 저장
        print("\n💾 5단계: 결과 저장")
        self._save_final_results(experimental_data, consistency_stats, rcbd_report)
        
        end_time = datetime.now()
        duration = end_time - start_time
        self._log(f"실험 완료: {end_time}, 소요시간: {duration}")
        
        # 종합 결과
        final_results = {
            'experimental_data': experimental_data,
            'consistency_statistics': consistency_stats,
            'rcbd_report': rcbd_report,
            'experiment_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': str(duration),
                'n_responses': n_responses,
                'temperature': temperature,
                'total_prompts': len(experimental_data),
                'total_responses': len(experimental_data) * n_responses
            },
            'experiment_log': self.experiment_log
        }
        
        print(f"\n✅ 실험 완료! 소요시간: {duration}")
        print(f"📂 결과 저장 위치: {self.output_dir}")
        
        return final_results
    
    def run_quick_test(self, n_responses=3, n_questions_per_category=3, save_intermediate=True):
        """
        빠른 테스트 실행 (일부 질문만 사용)
        
        Args:
            n_responses (int): 각 프롬프트당 수집할 응답 수
            n_questions_per_category (int): 카테고리당 사용할 질문 수
            save_intermediate (bool): 중간 결과 저장 여부
            
        Returns:
            dict: 테스트 결과
        """
        print("🧪 RCBD 실험 빠른 테스트")
        print("="*50)
        
        # 질문 수 제한
        test_questions = {}
        for category, questions in QUESTION_BLOCKS.items():
            test_questions[category] = questions[:n_questions_per_category]
        
        print(f"테스트 설정:")
        print(f"  - 카테고리당 질문 수: {n_questions_per_category}")
        print(f"  - 프롬프트당 응답 수: {n_responses}")
        print(f"  - 총 프롬프트 수: {len(test_questions) * n_questions_per_category * 3}")
        
        # 축소된 데이터로 실험 실행
        self.output_dir = os.path.join(self.output_dir, "quick_test")
        os.makedirs(self.output_dir, exist_ok=True)
        
        start_time = datetime.now()
        self._log(f"빠른 테스트 시작: {start_time}")
        
        # 1단계: 프롬프트 생성
        print("\n📝 1단계: 프레이밍 프롬프트 생성")
        experimental_data = self.prompt_generator.batch_generate_framings(test_questions)
        
        if save_intermediate:
            self._save_json(experimental_data, "01_test_prompts_generated.json")
        
        self._log(f"테스트 프롬프트 생성 완료: {len(experimental_data)}개")
        
        # 2단계: 응답 수집
        print(f"\n💬 2단계: AI 응답 수집 (각 프롬프트당 {n_responses}개)")
        experimental_data = self.response_collector.collect_responses_for_experiment(
            experimental_data, n_responses=n_responses, temperature=0.7
        )
        
        if save_intermediate:
            self._save_json(experimental_data, "02_test_responses_collected.json")
        
        self._log(f"테스트 응답 수집 완료: 총 {len(experimental_data) * n_responses}개 응답")
        
        # 3단계: 일관성 분석
        print("\n📊 3단계: 응답 일관성 분석")
        experimental_data = self.consistency_analyzer.analyze_consistency(experimental_data)
        
        if save_intermediate:
            self._save_json(experimental_data, "03_test_consistency_analyzed.json")
        
        # 일관성 통계 요약
        consistency_stats = self.consistency_analyzer.compute_consistency_statistics(experimental_data)
        self.consistency_analyzer.print_consistency_summary(consistency_stats)
        
        # 4단계: RCBD 분석
        print("\n📈 4단계: RCBD 통계 분석")
        rcbd_report = self.rcbd_analyzer.generate_comprehensive_report(experimental_data)
        
        # 5단계: 결과 저장
        if save_intermediate:
            print("\n💾 5단계: 최종 결과 저장")
            self._save_final_results(experimental_data, consistency_stats, rcbd_report)
        
        end_time = datetime.now()
        duration = end_time - start_time
        self._log(f"빠른 테스트 완료: {end_time}, 소요시간: {duration}")
        
        print(f"\n✅ 빠른 테스트 완료! 소요시간: {duration}")
        print(f"📂 결과 저장 위치: {self.output_dir}")
        
        return {
            'experimental_data': experimental_data,
            'consistency_statistics': consistency_stats,
            'rcbd_report': rcbd_report,
            'test_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration': str(duration),
                'n_responses': n_responses,
                'n_questions_per_category': n_questions_per_category,
                'total_prompts': len(experimental_data),
                'total_responses': len(experimental_data) * n_responses
            }
        }
    
    def _save_json(self, data, filename):
        """JSON 형태로 데이터 저장"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  → {filename} 저장 완료")
    
    def _save_final_results(self, experimental_data, consistency_stats, rcbd_report):
        """최종 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 실험 데이터 CSV
        df = pd.DataFrame(experimental_data)
        csv_path = os.path.join(self.output_dir, f"rcbd_experiment_data_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 2. RCBD 분석 결과 DataFrame
        rcbd_df = rcbd_report['dataframe']
        rcbd_csv_path = os.path.join(self.output_dir, f"rcbd_analysis_data_{timestamp}.csv")
        rcbd_df.to_csv(rcbd_csv_path, index=False, encoding='utf-8-sig')
        
        # 3. 요약 통계 JSON (JSON 직렬화 가능하도록 변환)
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
            'consistency_statistics': convert_to_serializable({str(k): v for k, v in consistency_stats.items()}),
            'rcbd_summary': convert_to_serializable(rcbd_report['summary']),
            'experiment_summary': {
                'total_conditions': len(set((d['category'], d['framing_level']) for d in experimental_data)),
                'avg_bert_multilingual_similarity': float(rcbd_df['bert_multilingual_similarity'].mean()) if 'bert_multilingual_similarity' in rcbd_df.columns else None
            }
        }
        
        summary_path = os.path.join(self.output_dir, f"experiment_summary_{timestamp}.json")
        self._save_json(summary_data, f"experiment_summary_{timestamp}.json")
        
        print(f"  → 실험 데이터: {csv_path}")
        print(f"  → RCBD 분석: {rcbd_csv_path}")
        print(f"  → 요약 통계: {summary_path}")
    
    def _log(self, message):
        """실험 로그 기록"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}"
        self.experiment_log.append(log_entry)
        print(f"📋 {message}")

# 실행 예시
if __name__ == "__main__":    
    # 실험 실행
    experiment = RCBDExperiment(OPENAI_API_KEY)
    
    # 빠른 테스트 (개발/디버깅용)
    results = experiment.run_quick_test()
    
    # 전체 실험 실행
    # results = experiment.run_complete_experiment(n_responses=5) 