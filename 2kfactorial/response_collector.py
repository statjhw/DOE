"""
2^5 Factorial Design용 AI 응답 수집기
다양한 모델과 프롬프트 조합으로부터 응답 수집
"""

import openai
import time
import random
from typing import Dict, List

class FactorialResponseCollector:
    def __init__(self, api_key: str, api_delay: float = 1.0):
        """
        응답 수집기 초기화
        
        Args:
            api_key (str): OpenAI API 키
            api_delay (float): API 호출 간 대기시간 (초)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.api_delay = api_delay
        
    def collect_single_response(self, prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """
        단일 프롬프트에 대한 응답 수집
        
        Args:
            prompt (str): 입력 프롬프트
            model (str): 사용할 모델
            temperature (float): 응답 다양성
            max_tokens (int): 최대 토큰 수
            
        Returns:
            str: AI 응답
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"   ⚠️ API 호출 오류: {e}")
            return f"[ERROR: {e}]"
    
    def collect_multiple_responses(self, prompt: str, model: str, n_responses: int = 3, temperature: float = 0.7) -> List[str]:
        """
        동일한 프롬프트에 대해 여러 응답 수집
        
        Args:
            prompt (str): 입력 프롬프트
            model (str): 사용할 모델
            n_responses (int): 수집할 응답 수
            temperature (float): 응답 다양성
            
        Returns:
            List[str]: 수집된 응답들
        """
        responses = []
        
        for i in range(n_responses):
            response = self.collect_single_response(prompt, model, temperature)
            responses.append(response)
            
            # API 호출 간 대기 (마지막 호출 제외)
            if i < n_responses - 1:
                time.sleep(self.api_delay)
        
        print(f"  → {len(responses)}개 응답 수집 완료")
        return responses
    
    def collect_responses_for_condition(self, condition: Dict, n_responses: int = 3, temperature: float = 0.7) -> Dict:
        """
        특정 조건에 대한 AI 응답 수집
        
        Args:
            condition (Dict): 실험 조건 (생성된 프롬프트 포함)
            n_responses (int): 수집할 응답 수
            temperature (float): 응답 생성 온도
            
        Returns:
            Dict: 응답이 포함된 조건 데이터
        """
        # 프롬프트와 모델 정보 추출
        prompt = condition['generated_prompt']
        model = condition['factor_combination']['model']
        
        print(f"  조건 ID: {condition['condition_id']}")
        print(f"  카테고리: {condition['category']}")
        print(f"  요인 조합: {', '.join([f'{k}:{v}' for k, v in condition['factor_combination'].items()])}")
        
        # 응답 수집
        responses = self.collect_multiple_responses(
            prompt=prompt,
            model=model, 
            n_responses=n_responses,
            temperature=temperature
        )
        
        # 결과 저장
        condition.update({
            'responses': responses,
            'n_responses': len(responses),
            'model_used': model,
            'temperature_used': temperature
        })
        
        return condition
    
    def collect_responses_for_experiment(self, experimental_data: List[Dict], n_responses: int = 3, temperature: float = 0.7) -> List[Dict]:
        """
        전체 실험에 대해 응답 수집
        
        Args:
            experimental_data (List[Dict]): 프롬프트가 포함된 실험 데이터
            n_responses (int): 각 조건당 수집할 응답 수
            temperature (float): 응답 다양성
            
        Returns:
            List[Dict]: 응답이 추가된 실험 데이터
        """
        print(f"\n💬 AI 응답 수집 시작 (각 조건당 {n_responses}개)")
        print("=" * 60)
        
        total_api_calls = len(experimental_data) * n_responses
        print(f"예상 API 호출 수: {total_api_calls}")
        print(f"예상 소요시간: {total_api_calls * self.api_delay / 60:.1f}분")
        
        for i, condition in enumerate(experimental_data):
            print(f"\n조건 {i+1}/{len(experimental_data)} 처리 중...")
            
            # 응답 수집
            condition = self.collect_responses_for_condition(condition, n_responses, temperature)
        
        print(f"\n✅ 응답 수집 완료!")
        print(f"   총 조건 수: {len(experimental_data)}")
        print(f"   총 응답 수: {len(experimental_data) * n_responses}")
        
        return experimental_data
    
    def collect_by_model_batch(self, experimental_data: List[Dict], n_responses: int = 3, temperature: float = 0.7) -> List[Dict]:
        """
        모델별로 묶어서 응답 수집 (효율성 향상)
        
        Args:
            experimental_data (List[Dict]): 실험 데이터
            n_responses (int): 각 조건당 응답 수
            temperature (float): 응답 다양성
            
        Returns:
            List[Dict]: 응답이 추가된 실험 데이터
        """
        print(f"\n💬 모델별 배치 응답 수집 시작")
        print("=" * 60)
        
        # 모델별로 그룹화
        model_groups = {}
        for condition in experimental_data:
            model = condition['factor_combination']['model']  # 수정: 올바른 경로
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(condition)
        
        print(f"모델별 조건 수:")
        for model, conditions in model_groups.items():
            print(f"  • {model}: {len(conditions)}개 조건")
        
        # 모델별로 순차 처리
        all_processed = []
        
        for model_name, conditions in model_groups.items():
            print(f"\n🤖 {model_name} 모델 응답 수집 중...")
            print("-" * 40)
            
            for i, condition in enumerate(conditions):
                print(f"  {i+1}/{len(conditions)}: 조건 {condition['condition_id']}")
                condition = self.collect_responses_for_condition(condition, n_responses, temperature)
                all_processed.append(condition)
        
        # 원래 순서로 정렬
        all_processed.sort(key=lambda x: x['condition_id'])
        
        print(f"\n✅ 모델별 배치 수집 완료!")
        return all_processed
    
    def _format_factors(self, factor_combination: Dict) -> str:
        """요인 조합을 읽기 쉬운 형태로 포맷"""
        formatted = []
        for factor, value in factor_combination.items():
            formatted.append(f"{factor}:{value}")
        return ", ".join(formatted)
    
    def estimate_cost(self, experimental_data: List[Dict], n_responses: int = 3) -> Dict:
        """
        실험 비용 추정
        
        Args:
            experimental_data (List[Dict]): 실험 데이터
            n_responses (int): 각 조건당 응답 수
            
        Returns:
            Dict: 비용 추정 정보
        """
        # 모델별 비용 (2024년 기준)
        model_costs = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},  # per 1K tokens
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006}   # per 1K tokens
        }
        
        model_counts = {}
        for condition in experimental_data:
            model = condition['factor_combination']['model']  # 수정: 올바른 경로
            model_counts[model] = model_counts.get(model, 0) + 1
        
        total_cost = 0
        cost_breakdown = {}
        
        for model, count in model_counts.items():
            if model in model_costs:
                # 대략적인 토큰 수 추정 (프롬프트 150 토큰, 응답 200 토큰)
                input_tokens = count * n_responses * 150
                output_tokens = count * n_responses * 200
                
                model_cost = (
                    (input_tokens / 1000) * model_costs[model]['input'] +
                    (output_tokens / 1000) * model_costs[model]['output']
                )
                
                cost_breakdown[model] = {
                    'conditions': count,
                    'total_responses': count * n_responses,
                    'estimated_cost': model_cost
                }
                total_cost += model_cost
        
        return {
            'total_estimated_cost': total_cost,
            'model_breakdown': cost_breakdown,
            'total_api_calls': len(experimental_data) * n_responses
        }

# 실행 예시
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("🧪 2^5 Factorial Response Collector 테스트")
    print("=" * 60)
    
    # 테스트용 샘플 조건
    test_condition = {
        'condition_id': 1,
        'category': '인성',
        'base_question': '어려움에 처한 친구를 도와야 하는 이유는 무엇인가?',
        'generated_prompt': '어려움에 처한 친구를 도와야 하는 이유는 무엇인가? 한국어로 답변해 주세요.',
        'factor_combination': {
            'prompt_language': 'korean',
            'model': 'gpt-4o-mini',
            'role_assignment': 'no_role',
            'context_provision': 'no_context',
            'explicitness': 'low'
        }
    }
    
    # 응답 수집기 초기화
    collector = FactorialResponseCollector(api_key, api_delay=0.5)
    
    # 비용 추정
    cost_info = collector.estimate_cost([test_condition], n_responses=2)
    print(f"\n💰 비용 추정:")
    print(f"   총 예상 비용: ${cost_info['total_estimated_cost']:.3f}")
    print(f"   총 API 호출: {cost_info['total_api_calls']}")
    
    print("\n✅ Response Collector 초기화 완료!") 