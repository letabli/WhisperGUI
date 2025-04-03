My laptop is on **Window 11**, **Cuda toolkit 12.1.1** and **Anaconda**.

Whisper_GUI.py 파일을 실행하기 위해서는 여러 라이브러리가 필요합니다. 코드를 살펴보니 OpenAI의 Whisper, 텍스트 요약 기능, 번역 기능 등이 포함된 Python 응용 프로그램입니다. 아래에 필요한 모든 종속성과 설치 방법을 안내해 드리겠습니다.

## 필요한 패키지 목록:

1. PyQt5 - GUI 프레임워크
2. Whisper - OpenAI의 음성 인식 모델
3. Torch - 딥러닝 프레임워크
4. NLTK - 자연어 처리 라이브러리
5. Sumy - 텍스트 요약 라이브러리
6. Googletrans - 번역 라이브러리
7. ReportLab - PDF 생성 라이브러리

## 설치 방법:

Anaconda를 사용 중이신 것으로 보아 conda 환경을 만들고 필요한 패키지를 설치하는 것이 좋겠습니다. 아래는 설치 명령어입니다:

```bash
# 1. 새로운 conda 환경 생성
conda create -n whisper_env python=3.9 -y
conda activate whisper_env

# 2. PyQt5 설치
conda install -c anaconda pyqt -y

# 3. Torch 설치 (CUDA 지원)
# CUDA 12.1.1이 설치되어 있다고 하셨으므로 호환되는 torch 버전 설치
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. OpenAI Whisper 설치
pip install openai-whisper

# 5. NLTK 설치
conda install -c anaconda nltk -y

# 6. Sumy 설치
pip install sumy

# 7. Googletrans 설치 (참고: 기본 googletrans는 종종 문제가 발생하므로 안정적인 버전 설치)
pip install googletrans==4.0.0-rc1

# 8. ReportLab 설치
pip install reportlab

# 9. NLTK 데이터 다운로드 (코드에서 자동으로 다운로드되지만 미리 설치 가능)
python -c "import nltk; nltk.download('punkt')"
```

## 실행 방법:

설치가 완료되면 다음과 같이 프로그램을 실행하세요:

```bash
# 환경이 활성화되어 있는지 확인
conda activate whisper_env

# 프로그램 실행
python Whi_trans.py
```

## 가능한 문제점 및 해결 방법:

1. **FFmpeg 관련 오류**: Whisper는 오디오/비디오 파일 처리를 위해 FFmpeg가 필요합니다.
   ```bash
   # FFmpeg 설치
   conda install -c conda-forge ffmpeg -y
   ```

2. **Googletrans 관련 오류**: googletrans 라이브러리가 작동하지 않는 경우 대체 방법으로 다음을 시도해 볼 수 있습니다:
   ```bash
   pip uninstall googletrans
   pip install googletrans==3.1.0a0
   ```

3. **Sumy 종속성 오류**: sumy가 다른 패키지를 요구할 수 있습니다:
   ```bash
   pip install numpy scipy scikit-learn
   ```

4. **torch와 CUDA 호환성 문제**: 특정 torch 버전이 CUDA 12.1.1과 호환되지 않는 경우:
   - [PyTorch 공식 웹사이트](https://pytorch.org/get-started/locally/)에서 CUDA 12.1에 맞는 정확한 설치 명령을 확인하세요.
  
5. **PyQt5 missing** 오류
```bash
pip uninstall PyQt5
pip install PyQt5
```

6. **NLTK 데이터 누락** 오류
```bash
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import nltk; nltk.download('stopwords')"
```
