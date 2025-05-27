"""
2^4 Factorial Design 실험 설정 (context_provision 제거)
ANOVA 결과: context_provision은 p=0.982로 효과 없음
"""

from typing import Dict, List

# 기본 실험 설정
DEFAULT_EXPERIMENT_CONFIG = {
    'n_responses': 5,          # 조건당 응답 수 (3 → 5로 증가)
    'temperature': 0.7,        # 응답 생성 온도
    'max_tokens': 500,         # 최대 토큰 수
    'retry_attempts': 3,       # 실패시 재시도 횟수
    'delay_between_requests': 1 # 요청 간 지연 (초)
}

# 실험 모드별 설정 (context_provision 제거 → 2^4 = 16 조합)
EXPERIMENT_MODES = {
    'demo': {
        'mode_name': '데모 실험',
        'max_conditions': 12,  # 2^2 × 3 카테고리
        'factors_to_test': ['prompt_language', 'model'],
        'questions_per_category': 1,
        'estimated_cost_usd': 0.03  # 5개 응답으로 증가
    },
    'test': {
        'mode_name': '테스트 실험', 
        'max_conditions': 36,  # 2^3 × 3 카테고리
        'factors_to_test': ['prompt_language', 'model', 'role_assignment'],
        'questions_per_category': 1,
        'estimated_cost_usd': 0.08  # 5개 응답으로 증가
    },
    'representative': {
        'mode_name': '대표 설계 (최적화)',
        'max_conditions': 48,  # 2^4 × 3 카테고리 (context_provision 제거)
        'factors_to_test': ['all'],
        'questions_per_category': 1,
        'estimated_cost_usd': 0.32  # 5개 응답으로 증가
    },
    'full': {
        'mode_name': '전체 설계 (최적화)',
        'max_conditions': 336,  # 2^4 × 21 질문 (7개×3카테고리, context_provision 제거)
        'factors_to_test': ['all'],
        'questions_per_category': 'all',
        'estimated_cost_usd': 2.24  # 7개 질문 × 5개 응답
    }
}

# OpenAI 모델별 비용 (1000 토큰당 USD)
MODEL_COSTS = {
    'gpt-3.5-turbo': {
        'input': 0.0015,
        'output': 0.002
    },
    'gpt-4o-mini': {
        'input': 0.00015,
        'output': 0.0006
    }
}

def get_experiment_config(mode: str = 'demo') -> Dict:
    """
    실험 모드에 따른 설정 반환
    
    Args:
        mode (str): 실험 모드 ('demo', 'test', 'representative', 'full')
        
    Returns:
        Dict: 실험 설정
    """
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"지원하지 않는 실험 모드: {mode}. 가능한 모드: {list(EXPERIMENT_MODES.keys())}")
    
    mode_config = EXPERIMENT_MODES[mode].copy()
    base_config = DEFAULT_EXPERIMENT_CONFIG.copy()
    
    # 설정 병합
    config = {**base_config, **mode_config}
    config['mode'] = mode
    
    return config

def estimate_experiment_cost(mode: str, n_responses: int = 3) -> Dict:
    """
    실험 비용 추정
    
    Args:
        mode (str): 실험 모드
        n_responses (int): 조건당 응답 수
        
    Returns:
        Dict: 비용 추정 정보
    """
    config = get_experiment_config(mode)
    
    # 평균 토큰 수 추정
    avg_input_tokens = 150   # 프롬프트 평균 토큰
    avg_output_tokens = 300  # 응답 평균 토큰
    
    total_requests = config['max_conditions'] * n_responses
    
    # 모델별 비용 계산 (gpt-3.5-turbo와 gpt-4o-mini 50:50 가정)
    gpt35_cost = (avg_input_tokens * MODEL_COSTS['gpt-3.5-turbo']['input'] + 
                  avg_output_tokens * MODEL_COSTS['gpt-3.5-turbo']['output']) / 1000
    
    gpt4o_mini_cost = (avg_input_tokens * MODEL_COSTS['gpt-4o-mini']['input'] + 
                       avg_output_tokens * MODEL_COSTS['gpt-4o-mini']['output']) / 1000
    
    avg_cost_per_request = (gpt35_cost + gpt4o_mini_cost) / 2
    total_cost = total_requests * avg_cost_per_request
    
    return {
        'mode': mode,
        'total_conditions': config['max_conditions'],
        'total_requests': total_requests,
        'avg_cost_per_request': avg_cost_per_request,
        'estimated_total_cost': total_cost,
        'breakdown': {
            'gpt-3.5-turbo_cost_per_request': gpt35_cost,
            'gpt-4o-mini_cost_per_request': gpt4o_mini_cost,
            'requests_per_model': total_requests // 2
        }
    }

def print_experiment_info(mode: str):
    """실험 정보 출력"""
    config = get_experiment_config(mode)
    cost_info = estimate_experiment_cost(mode, config['n_responses'])
    
    print(f"\n🔬 {config['mode_name']} 정보")
    print("=" * 50)
    print(f"📊 실험 설계: 2^4 Factorial Design")
    print(f"📊 테스트 요인: {len(config['factors_to_test'])}개")
    
    if config['factors_to_test'] != ['all']:
        print(f"   - 요인: {', '.join(config['factors_to_test'])}")
    else:
        print(f"   - 모든 요인 (4개): prompt_language, model, role_assignment, explicitness")
        print(f"   - 제거된 요인: context_provision (p=0.982, 효과 없음)")
    
    print(f"📊 총 조건 수: {config['max_conditions']}")
    print(f"📊 조건당 응답 수: {config['n_responses']}")
    print(f"📊 총 API 요청: {cost_info['total_requests']}")
    print(f"💰 예상 비용: ${cost_info['estimated_total_cost']:.3f}")

if __name__ == "__main__":
    print("🧪 2^4 Factorial Design 실험 설정")
    print("=" * 60)
    
    for mode in EXPERIMENT_MODES.keys():
        print_experiment_info(mode)
        print()
    
    print("💡 추천 (ANOVA 최적화 기반):")
    print("   - 빠른 테스트: demo 모드")
    print("   - 요인 효과 확인: test 모드") 
    print("   - 효율적 분석: representative 모드 (50% 비용 절약)")
    print("   - 완전한 실험: full 모드 (50% 비용 절약)")
    print("   ⚡ context_provision 제거로 모든 모드 비용 50% 절약!") 