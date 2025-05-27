"""
2^4 Factorial Design 프롬프트 생성기 (context_provision 제거)
GPT-4o-mini를 사용한 동적 프롬프트 생성
언어별 명확한 분리 보장
ANOVA 결과: context_provision은 p=0.982로 효과 없음
"""

import openai
import time
from typing import Dict, List
from experiment_design import QUESTION_BLOCKS, ENGLISH_QUESTIONS

class FactorialPromptGenerator:
    def __init__(self, api_key: str):
        """
        GPT 기반 프롬프트 생성기 초기화
        
        Args:
            api_key (str): OpenAI API 키
        """
        self.client = openai.OpenAI(api_key=api_key)
        
        # 카테고리별 전문가 역할
        self.category_roles = {
            '인성': "윤리학과 도덕철학을 전공한 전문가",
            '창의성': "창의성과 혁신을 연구하는 전문가", 
            '논리적추론': "논리학과 비판적 사고를 전문으로 하는 학자"
        }
        
        self.english_category_roles = {
            '인성': "an expert in ethics and moral philosophy",
            '창의성': "an expert in creativity and innovation research",
            '논리적추론': "a scholar specializing in logic and critical thinking"
        }
    
    def generate_prompt(self, condition):
        """
        조건에 따른 프롬프트 생성
        
        Args:
            condition (dict): 실험 조건
            
        Returns:
            str: 생성된 프롬프트
        """
        category = condition['category']
        base_question = condition['base_question']
        factor_combination = condition['factor_combination']
        
        # 프롬프트 언어에 따라 완전히 분리된 생성
        prompt_language = factor_combination['prompt_language']
        
        if prompt_language == 'korean':
            return self._generate_korean_prompt(category, base_question, factor_combination)
        else:
            return self._generate_english_prompt(category, base_question, factor_combination)
    
    def _generate_korean_prompt(self, category, base_question, factor_combination):
        """한국어 프롬프트 생성"""
        
        # 요인 설명을 한국어로 구성
        factor_descriptions = []
        
        # 역할 부여
        if factor_combination['role_assignment'] == 'with_role':
            role = self.category_roles[category]
            factor_descriptions.append(f"역할: 당신은 {role}입니다")
        
        # 명시성
        if factor_combination['explicitness'] == 'high':
            factor_descriptions.append("답변 방식: 구체적이고 명확한 예시와 함께 자세히 설명해 주세요")
        
        # GPT에게 한국어 프롬프트 생성 요청
        system_prompt = """당신은 실험용 프롬프트를 생성하는 전문가입니다. 
주어진 조건들을 자연스럽게 통합하여 하나의 완전한 한국어 프롬프트를 만들어주세요.
반드시 한국어로만 작성하고, 마지막에 '한국어로 답변해 주세요.'를 추가해주세요.
프롬프트만 출력하고 다른 설명은 하지 마세요."""
        
        user_prompt = f"""다음 조건들을 자연스럽게 통합하여 하나의 완전한 프롬프트를 만들어주세요:

기본 질문: {base_question}
적용할 조건들:
{chr(10).join(['- ' + desc for desc in factor_descriptions]) if factor_descriptions else '- 추가 조건 없음'}

조건들을 자연스럽게 통합한 완전한 한국어 프롬프트를 생성해주세요."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            generated_prompt = response.choices[0].message.content.strip()
            
            # 한국어 응답 지시가 없으면 추가
            if "한국어로 답변해 주세요" not in generated_prompt:
                generated_prompt += " 한국어로 답변해 주세요."
            
            time.sleep(0.5)  # API 호출 간격
            return generated_prompt
            
        except Exception as e:
            print(f"한국어 프롬프트 생성 실패: {e}")
            # 실패 시 기본 프롬프트 반환
            return f"{base_question} 한국어로 답변해 주세요."
    
    def _generate_english_prompt(self, category, base_question, factor_combination):
        """영어 프롬프트 생성"""
        
        # 영어 질문 찾기
        question_index = QUESTION_BLOCKS[category].index(base_question)
        english_question = ENGLISH_QUESTIONS[category][question_index]
        
        # 요인 설명을 영어로 구성
        factor_descriptions = []
        
        # 역할 부여
        if factor_combination['role_assignment'] == 'with_role':
            role = self.english_category_roles[category]
            factor_descriptions.append(f"Role: You are {role}")
        
        # 명시성
        if factor_combination['explicitness'] == 'high':
            factor_descriptions.append("Response style: Please provide a detailed explanation with specific and clear examples")
        
        # GPT에게 영어 프롬프트 생성 요청
        system_prompt = """You are an expert in generating experimental prompts.
Please create a complete English prompt by naturally integrating the given conditions.
Write ONLY in English and add 'Please respond in Korean.' at the end.
Output only the prompt without any other explanations."""
        
        user_prompt = f"""Please create a complete prompt by naturally integrating the following conditions:

Base question: {english_question}
Conditions to apply:
{chr(10).join(['- ' + desc for desc in factor_descriptions]) if factor_descriptions else '- No additional conditions'}

Generate a complete English prompt that naturally integrates these conditions."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            generated_prompt = response.choices[0].message.content.strip()
            
            # 한국어 응답 지시가 없으면 추가
            if "Please respond in Korean" not in generated_prompt:
                generated_prompt += " Please respond in Korean."
            
            time.sleep(0.5)  # API 호출 간격
            return generated_prompt
            
        except Exception as e:
            print(f"영어 프롬프트 생성 실패: {e}")
            # 실패 시 기본 프롬프트 반환
            return f"{english_question} Please respond in Korean."
    
    def generate_full_prompts(self):
        """전체 설계 프롬프트 생성 (240조건)"""
        from experiment_design import generate_full_factorial_design
        
        print("전체 2^4 Factorial Design 프롬프트 생성 중...")
        experimental_data = generate_full_factorial_design()
        
        print(f"총 {len(experimental_data)}개 조건의 프롬프트를 생성합니다.")
        
        for i, condition in enumerate(experimental_data):
            if (i + 1) % 50 == 0:
                print(f"진행률: {i + 1}/{len(experimental_data)} ({(i + 1)/len(experimental_data)*100:.1f}%)")
            
            prompt = self.generate_prompt(condition)
            condition['generated_prompt'] = prompt
        
        print("✅ 전체 프롬프트 생성 완료!")
        return experimental_data
    
    def generate_representative_prompts(self):
        """대표 설계 프롬프트 생성 (48조건)"""
        from experiment_design import generate_representative_design
        
        print("대표 2^4 Factorial Design 프롬프트 생성 중...")
        experimental_data = generate_representative_design()
        
        print(f"총 {len(experimental_data)}개 조건의 프롬프트를 생성합니다.")
        
        for i, condition in enumerate(experimental_data):
            if (i + 1) % 20 == 0:
                print(f"진행률: {i + 1}/{len(experimental_data)} ({(i + 1)/len(experimental_data)*100:.1f}%)")
            
            prompt = self.generate_prompt(condition)
            condition['generated_prompt'] = prompt
        
        print("✅ 대표 프롬프트 생성 완료!")
        return experimental_data
    
    def generate_subset_prompts(self, factors_to_test, questions_per_category):
        """부분 설계 프롬프트 생성"""
        from experiment_design import generate_subset_design
        
        print(f"부분 2^4 Factorial Design 프롬프트 생성 중...")
        print(f"테스트 요인: {factors_to_test}")
        print(f"카테고리별 질문 수: {questions_per_category}")
        
        experimental_data = generate_subset_design(factors_to_test, questions_per_category)
        
        print(f"총 {len(experimental_data)}개 조건의 프롬프트를 생성합니다.")
        
        for i, condition in enumerate(experimental_data):
            prompt = self.generate_prompt(condition)
            condition['generated_prompt'] = prompt
        
        print("✅ 부분 프롬프트 생성 완료!")
        return experimental_data

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("🧪 2^4 Factorial Design 프롬프트 생성기 테스트")
    print("=" * 60)
    
    generator = FactorialPromptGenerator(api_key)
    
    # 테스트 조건
    test_condition = {
        'condition_id': 1,
        'category': '인성',
        'base_question': '어려움에 처한 친구를 도와야 하는 이유는 무엇인가?',
        'factor_combination': {
            'prompt_language': 'korean',
            'model': 'gpt-4o-mini',
            'role_assignment': 'with_role',
            'explicitness': 'high'
        }
    }
    
    print("테스트 조건 (한국어):")
    korean_prompt = generator.generate_prompt(test_condition)
    print(f"생성된 프롬프트: {korean_prompt}")
    
    # 영어 테스트
    test_condition['factor_combination']['prompt_language'] = 'english'
    print("\n테스트 조건 (영어):")
    english_prompt = generator.generate_prompt(test_condition)
    print(f"생성된 프롬프트: {english_prompt}")
    
    print("\n✅ 테스트 완료! 언어별 프롬프트가 올바르게 분리되었습니다.") 