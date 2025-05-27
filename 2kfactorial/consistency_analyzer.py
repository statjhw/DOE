"""
2^5 Factorial Design 실험을 위한 일관성 분석기
BERT Multilingual 기반 BERTScore를 사용하여 응답 일관성 측정
"""

import re
import numpy as np
from bert_score import score
from itertools import combinations
from typing import List, Dict

class ConsistencyAnalyzer:
    def __init__(self, model_type='bert-base-multilingual-cased'):
        """
        일관성 분석기 초기화
        
        Args:
            model_type (str): BERT Multilingual 모델명 (한국어 지원 다국어 모델)
        """
        self.model_type = model_type
        print(f"BERT Multilingual 기반 일관성 분석기 초기화: {model_type}")
        
    def clean_text(self, text):
        """
        텍스트 전처리 (개행, 강조, 리스트 기호 등 제거)
        
        Args:
            text (str): 원본 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        text = text.replace('\n', ' ')
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **강조** 제거
        text = re.sub(r'\d+\.\s*', '', text)           # 숫자 리스트 제거
        text = re.sub(r'\s+', ' ', text)               # 중복 공백 제거
        return text.strip()
        
    def compute_bert_multilingual_similarity(self, response_list):
        """
        BERT Multilingual 기반 BERTScore를 사용한 응답 간 평균 유사도 계산
        
        Args:
            response_list (list): AI 응답들의 리스트
            
        Returns:
            float: 평균 BERT Multilingual BERTScore F1 (None if error)
        """
        if len(response_list) < 2:
            return None
            
        try:
            # 텍스트 전처리
            cleaned_responses = [self.clean_text(resp) for resp in response_list]
            
            # 모든 응답 쌍에 대해 BERTScore 계산
            similarities = []
            
            for resp1, resp2 in combinations(cleaned_responses, 2):
                # BERT Multilingual을 사용한 BERTScore 계산
                P, R, F1 = score(
                    [resp2], [resp1], 
                    model_type=self.model_type,
                    lang="ko", 
                    verbose=False
                )
                similarities.append(F1.item())
            
            # 모든 쌍의 평균 유사도 반환
            return np.mean(similarities)
            
        except Exception as e:
            print(f"BERT Multilingual BERTScore 계산 오류: {e}")
            return None
    
    def analyze_consistency_for_experiment(self, experimental_data: List[Dict]) -> List[Dict]:
        """
        2^5 Factorial Design 실험 데이터의 모든 응답에 대해 BERT Multilingual 기반 일관성 분석
        
        Args:
            experimental_data (list): 응답이 포함된 실험 데이터
            
        Returns:
            list: 일관성 점수가 추가된 실험 데이터
        """
        print("BERT Multilingual 기반 일관성 분석 시작...")
        
        for i, data in enumerate(experimental_data):
            print(f"분석 중 {i+1}/{len(experimental_data)}: "
                  f"조건 {data['condition_id']} - {data['category']}")
            
            responses = data.get('responses', [])
            
            if len(responses) >= 2:
                # BERT Multilingual BERTScore 계산
                bert_multilingual_similarity = self.compute_bert_multilingual_similarity(responses)
                data['bert_multilingual_similarity'] = bert_multilingual_similarity
                
                print(f"  → BERT Multilingual 유사도: {bert_multilingual_similarity:.4f}")
            else:
                data['bert_multilingual_similarity'] = None
                print(f"  → 응답 부족 (응답 수: {len(responses)})")
        
        print("BERT Multilingual 기반 일관성 분석 완료!")
        return experimental_data
    
    def compute_consistency_statistics(self, experimental_data: List[Dict]) -> Dict:
        """
        2^5 Factorial Design 조건별 BERT Multilingual 유사도 통계 계산
        
        Args:
            experimental_data (list): 일관성 점수가 포함된 실험 데이터
            
        Returns:
            dict: 조건별 통계 요약
        """
        # 요인별 데이터 그룹화
        factor_groups = {}
        
        for data in experimental_data:
            if data.get('bert_multilingual_similarity') is not None:
                # 각 요인별로 그룹화
                for factor_name, factor_value in data['factor_combination'].items():
                    if factor_name not in factor_groups:
                        factor_groups[factor_name] = {}
                    
                    if factor_value not in factor_groups[factor_name]:
                        factor_groups[factor_name][factor_value] = []
                    
                    factor_groups[factor_name][factor_value].append(data['bert_multilingual_similarity'])
                
                # 카테고리별로도 그룹화
                category = data['category']
                if 'category' not in factor_groups:
                    factor_groups['category'] = {}
                
                if category not in factor_groups['category']:
                    factor_groups['category'][category] = []
                
                factor_groups['category'][category].append(data['bert_multilingual_similarity'])
        
        # 통계 계산
        statistics = {}
        for factor_name, factor_values in factor_groups.items():
            statistics[factor_name] = {}
            
            for factor_value, similarities in factor_values.items():
                if similarities:
                    similarities_array = np.array(similarities)
                    statistics[factor_name][factor_value] = {
                        'n_samples': len(similarities),
                        'mean': similarities_array.mean(),
                        'std': similarities_array.std(),
                        'min': similarities_array.min(),
                        'max': similarities_array.max()
                    }
        
        return statistics
    
    def print_consistency_summary(self, statistics: Dict):
        """
        2^5 Factorial Design BERT Multilingual 기반 일관성 분석 결과 요약 출력
        
        Args:
            statistics (dict): 요인별 통계 데이터
        """
        print("\n" + "="*80)
        print("2^5 Factorial Design BERT Multilingual 일관성 분석 결과")
        print("="*80)
        
        for factor_name, factor_values in statistics.items():
            print(f"\n📊 요인: {factor_name}")
            print("-" * 60)
            
            for factor_value, stats in factor_values.items():
                print(f"\n  🎯 {factor_value}")
                print(f"     샘플 수: {stats['n_samples']}")
                print(f"     BERT Multilingual 유사도: {stats['mean']:.4f} (±{stats['std']:.4f})")
                print(f"     범위: {stats['min']:.4f} ~ {stats['max']:.4f}")
        
        # 전체 요약
        print(f"\n📈 요인별 효과 크기 (최대값 - 최소값)")
        print("-" * 60)
        
        for factor_name, factor_values in statistics.items():
            if len(factor_values) >= 2:
                means = [stats['mean'] for stats in factor_values.values()]
                effect_size = max(means) - min(means)
                print(f"  • {factor_name}: {effect_size:.4f}")

if __name__ == "__main__":
    print("🧪 2^5 Factorial Design 일관성 분석기 테스트")
    print("=" * 60)
    
    # 테스트용 샘플 데이터
    sample_data = [
        {
            'condition_id': 1,
            'category': '인성',
            'factor_combination': {
                'prompt_language': 'korean',
                'model': 'gpt-4o-mini',
                'role_assignment': 'with_role',
                'context_provision': 'no_context',
                'explicitness': 'low'
            },
            'responses': [
                '친구를 돕는 것은 인간의 기본적인 도덕적 의무입니다.',
                '어려운 상황에 있는 친구를 도와주는 것은 우정의 핵심이며, 상호부조의 정신을 보여줍니다.',
                '타인을 돕는 행위는 공감능력을 발휘하는 것이며, 사회적 유대감을 강화합니다.'
            ]
        }
    ]
    
    analyzer = ConsistencyAnalyzer()
    
    # 일관성 분석
    analyzed_data = analyzer.analyze_consistency_for_experiment(sample_data)
    
    # 통계 계산
    stats = analyzer.compute_consistency_statistics(analyzed_data)
    
    # 결과 출력
    analyzer.print_consistency_summary(stats)
    
    print("\n✅ 일관성 분석기 테스트 완료!") 