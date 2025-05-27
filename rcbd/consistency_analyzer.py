"""
RCBD 실험을 위한 일관성 분석기
BERT Multilingual 기반 BERTScore를 사용하여 응답 일관성 측정
"""

import re
import numpy as np
from bert_score import score
from itertools import combinations

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
    
    def analyze_consistency(self, experimental_data):
        """
        실험 데이터의 모든 응답에 대해 BERT Multilingual 기반 일관성 분석
        
        Args:
            experimental_data (list): 응답이 포함된 실험 데이터
            
        Returns:
            list: 일관성 점수가 추가된 실험 데이터
        """
        print("BERT Multilingual 기반 일관성 분석 시작...")
        
        for i, data in enumerate(experimental_data):
            print(f"분석 중 {i+1}/{len(experimental_data)}: "
                  f"{data['category']} × {data['framing_level']}")
            
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
    
    def compute_consistency_statistics(self, experimental_data):
        """
        조건별 BERT Multilingual 유사도 통계 계산
        
        Args:
            experimental_data (list): 일관성 점수가 포함된 실험 데이터
            
        Returns:
            dict: 조건별 통계 요약
        """
        # 조건별 데이터 그룹화
        conditions = {}
        for data in experimental_data:
            category = data['category']
            framing = data['framing_level']
            
            key = (category, framing)
            if key not in conditions:
                conditions[key] = {
                    'bert_multilingual_similarities': []
                }
            
            if data['bert_multilingual_similarity'] is not None:
                conditions[key]['bert_multilingual_similarities'].append(data['bert_multilingual_similarity'])
        
        # 통계 계산
        statistics = {}
        for (category, framing), values in conditions.items():
            stats = {
                'category': category,
                'framing_level': framing,
                'n_samples': len(values['bert_multilingual_similarities'])
            }
            
            # BERT Multilingual 유사도 통계
            if values['bert_multilingual_similarities']:
                bert_multilingual_array = np.array(values['bert_multilingual_similarities'])
                stats['bert_multilingual_mean'] = bert_multilingual_array.mean()
                stats['bert_multilingual_std'] = bert_multilingual_array.std()
                stats['bert_multilingual_min'] = bert_multilingual_array.min()
                stats['bert_multilingual_max'] = bert_multilingual_array.max()
            
            statistics[(category, framing)] = stats
        
        return statistics
    
    def print_consistency_summary(self, statistics):
        """
        BERT Multilingual 기반 일관성 분석 결과 요약 출력
        
        Args:
            statistics (dict): 조건별 통계 데이터
        """
        print("\n" + "="*80)
        print("BERT Multilingual 기반 일관성 분석 결과 요약")
        print("="*80)
        
        # 카테고리별 그룹화
        categories = {}
        for (category, framing), stats in statistics.items():
            if category not in categories:
                categories[category] = {}
            categories[category][framing] = stats
        
        for category, framings in categories.items():
            print(f"\n📂 카테고리: {category}")
            print("-" * 60)
            
            for framing, stats in framings.items():
                print(f"\n  🎯 프레이밍: {framing}")
                print(f"     샘플 수: {stats['n_samples']}")
                
                if 'bert_multilingual_mean' in stats:
                    print(f"     BERT Multilingual 유사도: {stats['bert_multilingual_mean']:.4f} "
                          f"(±{stats['bert_multilingual_std']:.4f})")
        
        # 전체 프레이밍별 평균
        print(f"\n📊 프레이밍별 전체 평균")
        print("-" * 60)
        
        framing_summary = {}
        for (category, framing), stats in statistics.items():
            if framing not in framing_summary:
                framing_summary[framing] = {
                    'bert_multilingual_scores': []
                }
            
            if 'bert_multilingual_mean' in stats:
                framing_summary[framing]['bert_multilingual_scores'].append(stats['bert_multilingual_mean'])
        
        for framing, scores in framing_summary.items():
            print(f"\n  🎯 {framing}")
            
            if scores['bert_multilingual_scores']:
                bert_multilingual_avg = np.mean(scores['bert_multilingual_scores'])
                print(f"     평균 BERT Multilingual 유사도: {bert_multilingual_avg:.4f}")
    
    # 레거시 메서드들 (하위 호환성을 위해 bert_multilingual_similarity로 매핑)
    def compute_cosine_similarity(self, response_list):
        """하위 호환성을 위한 메서드 - BERT Multilingual 유사도 반환"""
        return self.compute_bert_multilingual_similarity(response_list)
    
    def compute_bertscore(self, response_list):
        """하위 호환성을 위한 메서드 - BERT Multilingual 유사도 반환"""
        return self.compute_bert_multilingual_similarity(response_list)
    
    def compute_kobert_similarity(self, response_list):
        """하위 호환성을 위한 메서드 - BERT Multilingual 유사도 반환"""
        return self.compute_bert_multilingual_similarity(response_list)
    
    def compute_xlm_roberta_similarity(self, response_list):
        """하위 호환성을 위한 메서드 - BERT Multilingual 유사도 반환"""
        return self.compute_bert_multilingual_similarity(response_list) 