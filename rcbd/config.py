"""
RCBD 실험 설정 파일
"""

import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4o-mini"

# 실험 파라미터
DEFAULT_EXPERIMENT_PARAMS = {
    'n_responses': 4,          # 각 프롬프트당 수집할 응답 수 (351개 데이터 포인트용)
    'temperature': 0.7,        # 응답 다양성 조절
    'max_tokens': 500,         # 최대 토큰 수
    'api_delay': 1.0,          # API 호출 간 대기시간 (초)
}

# 분석 설정
BERT_MULTILINGUAL_MODEL = 'bert-base-multilingual-cased'  # BERT 다국어 모델 (한국어 지원)
BERTSCORE_LANG = 'ko'                                     # BERTScore 언어 설정

# 출력 설정
OUTPUT_DIR = "results"
SAVE_INTERMEDIATE = True

# 실험 조건 확인
def validate_config():
    """설정 유효성 검사"""
    if not OPENAI_API_KEY:
        print("⚠️  경고: OpenAI API 키가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하거나 직접 입력하세요.")
        return False
    return True

# 실험 요약 정보
EXPERIMENT_INFO = {
    'title': 'RCBD 감정 프레이밍 실험',
    'objective': '감정 프레이밍에 따른 AI 응답 일관성 변화 분석',
    'design': 'Randomized Complete Block Design (RCBD)',
    'factors': {
        'treatment': {
            'name': '프레이밍 수준',
            'levels': ['중립적', '정서적', '자극적'],
            'description': '질문의 감정적 표현 강도'
        },
        'block': {
            'name': '질문 카테고리',
            'levels': ['인성', '창의성', '논리적추론'],
            'description': '질문의 주제 영역'
        }
    },
    'response_variables': {
        'primary': 'bert_multilingual_similarity',
        'secondary': None,
        'description': 'BERT Multilingual 기반 BERTScore를 사용한 AI 응답 간 일관성 측정값'
    }
} 