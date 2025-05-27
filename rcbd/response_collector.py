"""
RCBD 실험을 위한 응답 수집기
프레이밍된 프롬프트에 대해 AI 응답을 여러 번 수집하여 일관성 측정을 위한 데이터 생성
"""

from openai import OpenAI
import time

class ResponseCollector:
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def collect_single_response(self, prompt, temperature=0.7):
        """
        단일 프롬프트에 대한 응답 수집
        
        Args:
            prompt (str): 입력 프롬프트
            temperature (float): 응답 다양성 조절 파라미터
            
        Returns:
            str: AI 응답
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip().replace("\n", " ")
            
        except Exception as e:
            print(f"응답 수집 오류: {e}")
            return f"[ERROR] {e}"
    
    def collect_multiple_responses(self, prompt, n_responses=5, temperature=0.7):
        """
        동일한 프롬프트에 대해 여러 번의 응답 수집
        
        Args:
            prompt (str): 입력 프롬프트
            n_responses (int): 수집할 응답 개수
            temperature (float): 응답 다양성 조절 파라미터
            
        Returns:
            list: AI 응답들의 리스트
        """
        responses = []
        
        for i in range(n_responses):
            response = self.collect_single_response(prompt, temperature)
            responses.append(response)
            time.sleep(1)  # API 호출 간격 조절
            
        return responses
    
    def collect_responses_for_experiment(self, experimental_data, n_responses=5, temperature=0.7):
        """
        실험 데이터의 모든 프롬프트에 대해 응답 수집
        
        Args:
            experimental_data (list): 실험 데이터 리스트
            n_responses (int): 각 프롬프트당 수집할 응답 개수
            temperature (float): 응답 다양성 조절 파라미터
            
        Returns:
            list: 응답이 추가된 실험 데이터
        """
        total_prompts = len(experimental_data)
        
        for i, data in enumerate(experimental_data):
            print(f"\n프롬프트 {i+1}/{total_prompts} 처리 중...")
            print(f"카테고리: {data['category']}, 프레이밍: {data['framing_level']}")
            print(f"프롬프트: {data['framed_prompt'][:100]}...")
            
            # 응답 수집
            responses = self.collect_multiple_responses(
                data['framed_prompt'], 
                n_responses=n_responses, 
                temperature=temperature
            )
            
            # 데이터에 응답 추가
            data['responses'] = responses
            data['n_responses'] = len(responses)
            
            print(f"  → {len(responses)}개 응답 수집 완료")
            
            # 프롬프트 간 대기
            if i < total_prompts - 1:
                time.sleep(2)
        
        return experimental_data
    
    def collect_responses_by_condition(self, experimental_data, n_responses=5, temperature=0.7):
        """
        조건별로 응답 수집 진행 상황을 보고
        
        Args:
            experimental_data (list): 실험 데이터 리스트
            n_responses (int): 각 프롬프트당 수집할 응답 개수
            temperature (float): 응답 다양성 조절 파라미터
            
        Returns:
            list: 응답이 추가된 실험 데이터
        """
        # 조건별 그룹화
        conditions = {}
        for data in experimental_data:
            key = (data['category'], data['framing_level'])
            if key not in conditions:
                conditions[key] = []
            conditions[key].append(data)
        
        print(f"전체 조건 수: {len(conditions)}")
        print(f"조건당 프롬프트 수: {len(list(conditions.values())[0])}")
        print(f"전체 프롬프트 수: {len(experimental_data)}")
        
        # 조건별 처리
        for i, ((category, framing), data_list) in enumerate(conditions.items()):
            print(f"\n=== 조건 {i+1}/{len(conditions)}: {category} × {framing} ===")
            
            for j, data in enumerate(data_list):
                print(f"  프롬프트 {j+1}/{len(data_list)}: {data['framed_prompt'][:80]}...")
                
                # 응답 수집
                responses = self.collect_multiple_responses(
                    data['framed_prompt'], 
                    n_responses=n_responses, 
                    temperature=temperature
                )
                
                # 데이터에 응답 추가
                data['responses'] = responses
                data['n_responses'] = len(responses)
                
                print(f"    → {len(responses)}개 응답 수집 완료")
                
                time.sleep(1)
        
        return experimental_data 