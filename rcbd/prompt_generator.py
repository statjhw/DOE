"""
RCBD 실험을 위한 프롬프트 생성기
각 질문에 대해 3가지 프레이밍 수준(중립적, 정서적, 자극적)의 프롬프트를 생성
"""

from openai import OpenAI
import time
from experiment_design import FRAMING_DEFINITIONS

class FramingPromptGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def generate_framed_prompt(self, base_question, framing_level):
        """
        기본 질문을 특정 프레이밍 수준에 맞게 변형
        
        Args:
            base_question (str): 기본 질문
            framing_level (str): 프레이밍 수준 ("중립적", "정서적", "자극적")
            
        Returns:
            str: 프레이밍이 적용된 질문
        """
        framing_info = FRAMING_DEFINITIONS[framing_level]
        
        system_prompt = f"""
        너는 "프롬프트 변형 전문가"다. 주어진 질문을 특정 감정 프레이밍 수준에 맞게 변형해야 한다.
        
        [프레이밍 수준: {framing_level}]
        설명: {framing_info['description']}
        
        특징:
        """ + "\n".join([f"- {char}" for char in framing_info['characteristics']]) + f"""
        
        [변형 규칙]
        1. 질문의 핵심 주제와 의미는 유지해야 함
        2. {framing_level} 프레이밍의 특징을 명확히 반영해야 함
        3. 한 문장으로 작성 (최대 2문장 허용)
        4. 자연스럽고 현실적인 표현 사용
        5. 따옴표, 번호, 부가 설명 없이 변형된 질문만 출력
        
        [주의사항]
        - 중립적: 감정 없이 객관적으로
        - 정서적: 온건한 감정과 개인적 관심 포함
        - 자극적: 강한 감정, 비판, 위기감 표현
        """
        
        user_prompt = f"""
        다음 질문을 '{framing_level}' 프레이밍 수준에 맞게 변형해주세요.
        
        원본 질문: {base_question}
        
        {framing_level} 프레이밍의 특징을 반영한 변형된 질문을 한 줄로 출력해주세요.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"프롬프트 생성 오류: {e}")
            return f"[ERROR] {e}"
    
    def generate_all_framings(self, base_question):
        """
        한 질문에 대해 모든 프레이밍 수준의 변형 생성
        
        Args:
            base_question (str): 기본 질문
            
        Returns:
            dict: 프레이밍 수준별 변형된 질문들
        """
        results = {}
        
        for framing in ["중립적", "정서적", "자극적"]:
            framed_prompt = self.generate_framed_prompt(base_question, framing)
            results[framing] = framed_prompt
            time.sleep(1)  # API 호출 간격 조절
            
        return results
    
    def batch_generate_framings(self, questions_dict):
        """
        여러 카테고리의 질문들에 대해 일괄 프레이밍 생성
        
        Args:
            questions_dict (dict): 카테고리별 질문 딕셔너리
            
        Returns:
            list: 실험 데이터 리스트 (category, question, framing, prompt)
        """
        experimental_data = []
        
        for category, questions in questions_dict.items():
            print(f"\n카테고리 '{category}' 처리 중...")
            
            for i, question in enumerate(questions):
                print(f"  질문 {i+1}/{len(questions)}: {question[:50]}...")
                
                framings = self.generate_all_framings(question)
                
                for framing_level, framed_prompt in framings.items():
                    experimental_data.append({
                        "category": category,
                        "base_question": question,
                        "framing_level": framing_level,
                        "framed_prompt": framed_prompt
                    })
                
                time.sleep(0.5)  # API 호출 간격
        
        return experimental_data 