# On-device LLM demo

## 🎯 프로젝트 개요

이 프로젝트는 한-영, 영-한 번역 작업이 가능한 LLM을 on-device 환경(주로 스마트폰)에서 구동하기 위해 Unsloth를 활용하여 양자화된 언어 모델을 한국어, 영어 간 교차 번역 작업에 파인튜닝하고 평가하는 과정을 구현한 것입니다.
이를 위해 한국어, 영어 교차 번역을 위해 OpenSubtitles, TED2020과 같은 공개 영어 데이터셋을 OpenAI Batch API를 통해 저렴하게 고품질로 번역하고, 데이터셋으로 활용합니다.

모델 훈련에는 Unsloth와 deepSpeed를 활용하여 4bit, 8bit, 16bit 모델에 대한 효율적인 fine-tuning을 진행하였습니다.

모델은 fine-tuning 과정에서 Chat-template 기반으로 훈련되어 스마트폰, 노트북, 가정용 데스크탑과 같은 on-deivce 환경에서 다운로드한 뒤, 모델을 로드하여 챗봇처럼 사용 가능한 오픈소스 어플리케이션 또는 프로젝트를 통해 활용할 수 있습니다.
(데모 영상이 README에 첨부되어 있습니다!)

## 📁 프로젝트 구조

```
OLM_PROJECTS/
├── 📁 scripts/                    # 훈련 및 모델 관련 스크립트
│   ├── 🐍 train.py               # 모델 훈련 스크립트
│   ├── 📜 train.sh               # 훈련 실행 스크립트
│   ├── ⚙️ ds_config.json         # DeepSpeed 설정
│   ├── 📁 model_ver1/            # 모델 버전 1 관련 파일
│   ├── 📁 model_ver2/            # 모델 버전 2 관련 파일
│   ├── 📁 dataset_ver1/          # 데이터셋 버전 1 처리 스크립트
│   └── 📁 dataset_ver2/          # 데이터셋 버전 2 처리 스크립트
├── 📁 llm_evaluation/            # 언어 모델 평가 도구
│   ├── 🐍 1. generation.py       # 텍스트 생성 평가
│   └── 📜 generation_*.sh        # 평가 실행 스크립트들 (GPU별)
├── 🐍 upload_model.py            # 모델 업로드 스크립트
├── 📓 upload_model.ipynb         # 모델 업로드 노트북
├── ⚙️ config.env.example         # 환경변수 설정 예제
├── 📋 requirements.txt           # Python 패키지 의존성
├── 🚫 .gitignore                 # Git 제외 파일 목록
└── 📖 README.md                  # 프로젝트 문서
```

## 🚀 주요 기능

### 1. 🏋️ 모델 훈련
- **train.py**: Unsloth 기반 언어 모델 파인튜닝
- **train.sh**: 훈련 프로세스 자동화
- **ds_config.json**: DeepSpeed 최적화 설정
- 다양한 모델 크기 및 아키텍처 지원

### 2. 📤 모델 업로드
- **upload_model.py**: Hugging Face Hub에 GGUF 형식으로 모델 업로드
- 지원 양자화: q4_k_m, q8_0, f16
- 자동 모델 카드 생성 및 메타데이터 관리

### 3. 📊 모델 평가
- **generation.py**: 텍스트 생성 품질 평가
- 다양한 generation 파라미터 테스트
- 멀티 GPU 지원으로 병렬 평가

### 4. 🔄 데이터 처리
- **dataset_ver1/**: OpenSubtitles 및 TED 데이터셋 처리
- **dataset_ver2/**: 다국어 번역 데이터셋 처리
- OpenAI Batch API를 활용한 대규모 번역 작업

## 📦 설치 및 설정

### 1. 저장소 클론 및 의존성 설치

```bash
git clone https://github.com/your_username/OLM_PROJECTS.git
cd OLM_PROJECTS
pip install -r requirements.txt
```

### 2. 환경변수 설정

⚠️ **중요**: API 키와 경로 설정이 필요합니다.

1. **환경변수 파일 생성**:
   ```bash
   cp config.env.example config.env
   ```

2. **API 키 설정** (`config.env` 파일 편집):
   ```bash
   # 실제 API 키로 변경
   OPENAI_API_KEY=sk-your-actual-openai-api-key
   HUGGINGFACE_TOKEN=hf_your-actual-huggingface-token
   
   # 모델 저장 경로 설정
   MODEL_BASE_PATH=/path/to/your/models
   
   # 데이터셋 이름 설정
   EVALUATION_DATASET=your_username/your_dataset_name
   ```

3. **환경변수 로드**:
   ```bash
   source config.env
   ```

### 🔧 주요 환경변수

| 변수명 | 설명 | 기본값 | 필수 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | OpenAI API 키 | - | ✅ |
| `HUGGINGFACE_TOKEN` | Hugging Face 토큰 | - | ✅ |
| `MODEL_BASE_PATH` | 모델 저장 경로 | `/path/to/your/models` | ✅ |
| `RESULTS_BASE_PATH` | 결과 저장 경로 | `./llm_evaluation/results` | ❌ |
| `EVALUATION_DATASET` | 평가용 데이터셋 | `your_username/your_dataset_name` | ✅ |
| `MAX_SEQ_LENGTH` | 최대 시퀀스 길이 | `4096` | ❌ |
| `MAX_NEW_TOKENS` | 최대 생성 토큰 수 | `4096` | ❌ |
| `TEMPERATURE` | 생성 온도 | `0.9` | ❌ |

## 🎯 사용법

### 모델 훈련
```bash
cd scripts
# 환경변수 로드
source ../config.env
# 훈련 실행
bash train.sh
```

### 모델 업로드
```bash
# 환경변수 로드
source config.env
# 모델 업로드
python upload_model.py --model_path="your_model_path" --hf_path="your_hf_repo"
```

### 모델 평가
```bash
cd llm_evaluation
# 환경변수 로드
source ../config.env
# 평가 실행 (GPU 1에서 실행)
bash generation_1.sh
```

### 데이터 처리
```bash
cd scripts/dataset_ver1
# 환경변수 로드
source ../../config.env
# Jupyter 노트북 실행
jupyter notebook
```

## ⚙️ 고급 설정

### DeepSpeed 설정
`scripts/ds_config.json`에서 DeepSpeed 파라미터를 조정할 수 있습니다:
- Zero 최적화 레벨
- 메모리 관리 설정
- 통신 백엔드 설정

### 훈련 파라미터
`scripts/train.py`에서 다음 파라미터들을 수정할 수 있습니다:
- 학습률 (learning rate)
- 배치 크기 (batch size)
- 에포크 수 (epochs)
- 모델 아키텍처
- 드롭아웃 비율

## 📊 데이터 관리

### 사용힌 데이터셋
- **OpenSubtitles**: 영화/TV 자막 데이터 (영어-한국어)
- **TED Talks**: 강연 스크립트 (다국어)
- **사용자 정의 번역 데이터셋**: 특정 도메인 데이터

### 데이터 처리 파이프라인
1. **데이터 수집**: API를 통한 자동 수집
2. **전처리**: 정리, 필터링, 정규화
3. **번역**: OpenAI Batch API 활용
4. **품질 검증**: 자동 품질 점수 계산
5. **최종 데이터셋 생성**: 모델 훈련용 형식으로 변환


## 🎬 데모 영상

해당 영상은 iOS 기반 환경에서 Chat Model을 테스트할 수 있는 오픈소스 애플리케이션인 LLMFarm (https://github.com/guinmoon/LLMFarm) 을 통해 구동되었습니다.

fine-tuning 된 모델의 시연을 위해, 개인 소유의 iPhone 16 pro (8GB RAM)을 사용하였습니다.


https://github.com/user-attachments/assets/a76bb425-dd37-4808-a24c-f8a00ef24780



## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.
