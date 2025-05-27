"""
2^4 Factorial Design 실험 설계 (context_provision 제거)
4개 요인 (2^4 = 16 조합) × 질문 카테고리

요인:
1. 프롬프트 언어 (한국어/영어) - 응답은 한국어로 고정
2. 모델 (gpt-3.5-turbo/gpt-4o-mini)  
3. 역할 부여 (있음/없음) - 카테고리별 전문가 역할
4. 명시성 (높음/낮음)

제거된 요인:
- 맥락 제공: ANOVA 결과 p=0.982로 효과 없음

카테고리: 인성, 창의성, 논리적추론 (분석 시 블록 또는 요인으로 처리)
프롬프트 생성: GPT-4o-mini로 동적 생성
"""

import itertools
from typing import Dict, List, Tuple

# 실험 요인 정의 (2^4 Factorial Design)
FACTORS = {
    'prompt_language': ['korean', 'english'],      # 프롬프트 언어
    'model': ['gpt-3.5-turbo', 'gpt-4o-mini'],    # 모델
    'role_assignment': ['with_role', 'no_role'],   # 역할 부여
    'explicitness': ['high', 'low']                # 명시성
    # context_provision 제거: ANOVA 결과 p=0.982로 효과 없음
}

# 질문 카테고리 (분석 시 블록 또는 6번째 요인으로 처리)
QUESTION_CATEGORIES = ['인성', '창의성', '논리적추론']

# 각 카테고리별 기본 질문들
QUESTION_BLOCKS = {
    '인성': [
        "어려움에 처한 친구를 도와야 하는 이유는 무엇인가?",
        "정의로운 사회를 만들기 위해 개인이 해야 할 일은 무엇인가?",
        "윤리적 딜레마 상황에서 올바른 선택을 하는 기준은 무엇인가?",
        "타인과의 갈등 상황에서 화해를 이루는 방법은 무엇인가?",
        "자신의 실수를 인정하고 책임지는 것의 중요성은 무엇인가?",
        "타인에 대한 배려와 공감이 중요한 이유는 무엇인가?",
        "도덕적 용기를 발휘해야 하는 상황과 그 방법은 무엇인가?"
    ],
    '창의성': [
        "새로운 아이디어를 생각해내는 가장 효과적인 방법은 무엇인가?",
        "창의적 문제해결을 위해 어떤 사고방식이 필요한가?",
        "혁신적인 솔루션을 개발하는 과정에서 중요한 요소는 무엇인가?",
        "기존의 관습을 뛰어넘는 발상을 하는 방법은 무엇인가?",
        "다양한 관점에서 문제를 바라보는 능력을 기르는 방법은 무엇인가?",
        "창의성을 저해하는 요소들을 극복하는 방법은 무엇인가?",
        "협업을 통해 창의적 아이디어를 발전시키는 방법은 무엇인가?"
    ],
    '논리적추론': [
        "복잡한 문제를 체계적으로 분석하는 방법은 무엇인가?",
        "논리적 결론에 도달하기 위해 필요한 사고 과정은 무엇인가?",
        "증거와 논리를 바탕으로 타당한 주장을 펼치는 방법은 무엇인가?",
        "다양한 정보를 종합하여 올바른 판단을 내리는 방법은 무엇인가?",
        "논리적 오류를 피하고 합리적 사고를 하는 방법은 무엇인가?",
        "가설을 설정하고 검증하는 체계적인 방법은 무엇인가?",
        "비판적 사고를 통해 정보의 신뢰성을 평가하는 방법은 무엇인가?"
    ]
}

# 영어 질문 번역
ENGLISH_QUESTIONS = {
    '인성': [
        "What are the reasons for helping a friend in difficulty?",
        "What should individuals do to create a just society?",
        "What are the criteria for making the right choice in ethical dilemma situations?",
        "What are the ways to achieve reconciliation in conflict situations with others?",
        "What is the importance of acknowledging and taking responsibility for one's mistakes?",
        "Why are consideration and empathy for others important?",
        "What are the situations where moral courage should be exercised and how to do it?"
    ],
    '창의성': [
        "What is the most effective way to come up with new ideas?",
        "What kind of mindset is needed for creative problem solving?",
        "What are the important factors in developing innovative solutions?",
        "What are the ways to think beyond existing conventions?",
        "What are the ways to develop the ability to look at problems from various perspectives?",
        "What are the ways to overcome factors that hinder creativity?",
        "What are the ways to develop creative ideas through collaboration?"
    ],
    '논리적추론': [
        "What are the methods for systematically analyzing complex problems?",
        "What thought processes are necessary to reach logical conclusions?",
        "What are the methods for presenting valid arguments based on evidence and logic?",
        "What are the ways to make correct judgments by synthesizing various information?",
        "What are the ways to avoid logical errors and engage in rational thinking?",
        "What are the systematic methods for setting up and verifying hypotheses?",
        "What are the methods for evaluating the reliability of information through critical thinking?"
    ]
}

def generate_all_factor_combinations() -> List[Dict]:
    """
    모든 요인 조합 생성 (2^4 = 16개)
    
    Returns:
        List[Dict]: 각 조합의 요인 설정들
    """
    factor_names = list(FACTORS.keys())
    factor_values = list(FACTORS.values())
    
    combinations = []
    for combo in itertools.product(*factor_values):
        combination = dict(zip(factor_names, combo))
        combinations.append(combination)
    
    return combinations

def generate_full_factorial_design() -> List[Dict]:
    """
    전체 2^4 Factorial Design 생성
    모든 요인 조합 × 모든 질문 (2^4 × 15 = 240조건)
    
    Returns:
        List[Dict]: 전체 실험 조건들
    """
    factor_combinations = generate_all_factor_combinations()
    experimental_conditions = []
    
    condition_id = 1
    for combination in factor_combinations:
        for category, questions in QUESTION_BLOCKS.items():
            for question in questions:
                condition = {
                    'condition_id': condition_id,
                    'category': category,
                    'base_question': question,
                    'factor_combination': combination
                }
                experimental_conditions.append(condition)
                condition_id += 1
    
    return experimental_conditions

def generate_representative_design() -> List[Dict]:
    """
    대표 질문 설계 생성 (분석 테스트용)
    각 카테고리의 첫 번째 질문만 사용 (2^4 × 3 = 48조건)
    
    Returns:
        List[Dict]: 대표 실험 조건들
    """
    factor_combinations = generate_all_factor_combinations()
    experimental_conditions = []
    
    # 각 카테고리에서 첫 번째 질문만 사용
    representative_questions = {
        category: questions[0] 
        for category, questions in QUESTION_BLOCKS.items()
    }
    
    condition_id = 1
    for combination in factor_combinations:
        for category, question in representative_questions.items():
            condition = {
                'condition_id': condition_id,
                'category': category,
                'base_question': question,
                'factor_combination': combination
            }
            experimental_conditions.append(condition)
            condition_id += 1
    
    return experimental_conditions

def generate_subset_design(factors_to_test: List[str], questions_per_category: int = None) -> List[Dict]:
    """
    특정 요인들만 테스트하는 부분 설계 생성
    
    Args:
        factors_to_test (List[str]): 테스트할 요인들
        questions_per_category (int): 카테고리별 질문 수 (None이면 모든 질문)
        
    Returns:
        List[Dict]: 부분 실험 조건들
    """
    # 테스트할 요인들만 추출
    test_factors = {k: v for k, v in FACTORS.items() if k in factors_to_test}
    
    # 나머지 요인들의 기본값
    factor_defaults = {
        'prompt_language': 'korean',
        'model': 'gpt-4o-mini',
        'role_assignment': 'no_role',
        'explicitness': 'low'
    }
    
    # 테스트 요인 조합 생성
    test_factor_names = list(test_factors.keys())
    test_factor_values = list(test_factors.values())
    
    experimental_conditions = []
    condition_id = 1
    
    for combo in itertools.product(*test_factor_values):
        # 전체 요인 조합 구성
        full_combination = factor_defaults.copy()
        test_combination = dict(zip(test_factor_names, combo))
        full_combination.update(test_combination)
        
        # 질문 선택
        for category, questions in QUESTION_BLOCKS.items():
            selected_questions = questions[:questions_per_category] if questions_per_category else questions
            
            for question in selected_questions:
                condition = {
                    'condition_id': condition_id,
                    'category': category,
                    'base_question': question,
                    'factor_combination': full_combination
                }
                experimental_conditions.append(condition)
                condition_id += 1
    
    return experimental_conditions

def get_design_summary():
    """실험 설계 요약 정보"""
    factor_combinations = generate_all_factor_combinations()
    full_design = generate_full_factorial_design()
    representative_design = generate_representative_design()
    
    return {
        'total_factors': len(FACTORS),
        'levels_per_factor': [len(levels) for levels in FACTORS.values()],
        'total_factor_combinations': len(factor_combinations),
        'question_categories': len(QUESTION_CATEGORIES),
        'questions_per_category': [len(questions) for questions in QUESTION_BLOCKS.values()],
        'total_questions': sum(len(questions) for questions in QUESTION_BLOCKS.values()),
        'full_design_conditions': len(full_design),
        'representative_design_conditions': len(representative_design),
        'factors': FACTORS,
        'design_info': {
            'full_design': f"2^4 factorial = {len(factor_combinations)} combinations × {sum(len(q) for q in QUESTION_BLOCKS.values())} total questions = {len(full_design)} conditions",
            'representative_design': f"2^4 factorial = {len(factor_combinations)} combinations × {len(QUESTION_CATEGORIES)} categories = {len(representative_design)} conditions"
        }
    }

if __name__ == "__main__":
    # 설계 요약 출력
    summary = get_design_summary()
    print("🔬 2^4 Factorial Design 실험 설계 요약")
    print("=" * 60)
    print(f"📊 요인 수: {summary['total_factors']}")
    print(f"📊 요인별 수준: {dict(zip(FACTORS.keys(), summary['levels_per_factor']))}")
    print(f"📊 총 요인 조합: {summary['total_factor_combinations']}")
    print(f"📊 질문 카테고리: {summary['question_categories']}")
    print(f"📊 카테고리별 질문 수: {dict(zip(QUESTION_CATEGORIES, summary['questions_per_category']))}")
    print(f"📊 총 질문 수: {summary['total_questions']}")
    print()
    print("🎯 실험 조건 수:")
    print(f"   - 전체 설계: {summary['full_design_conditions']}")
    print(f"   - 대표 설계: {summary['representative_design_conditions']}")
    print()
    print("📋 설계 상세:")
    print(f"   - 전체 설계: {summary['design_info']['full_design']}")
    print(f"   - 대표 설계: {summary['design_info']['representative_design']}")
    print()
    print("📝 고정 설정:")
    print("   - 프롬프트 생성: GPT-4o-mini 동적 생성")
    print("   - 역할 부여: 카테고리별 전문가 (윤리학자, 창의성 전문가, 논리학자)") 