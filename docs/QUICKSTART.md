# Pose Splatter Quick Start Guide

**목표**: 최소 시간 내 환경 구축 및 코드 검증

---

## 🚀 5분 Quick Start

### Step 1: 환경 생성 (2분)

```bash
# Conda 환경 생성
conda create -n pose-splatter python=3.10 -y
conda activate pose-splatter
```

### Step 2: PyTorch 설치 (3분)

```bash
# CUDA 11.8 버전
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Step 3: 필수 패키지 설치 (5분)

```bash
# 핵심 패키지
pip install gsplat torch-scatter zarr h5py opencv-python torchmetrics matplotlib Pillow tqdm joblib

# torch-scatter 실패 시
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Step 4: 검증 (30초)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from src.model import PoseSplatter; print('Model: OK')"
```

**성공하면 환경 구축 완료!** ✅

---

## 📋 체크리스트

설치 전:
- [ ] NVIDIA GPU 있음
- [ ] `nvidia-smi` 작동 확인
- [ ] Conda 설치됨

설치 후:
- [ ] `torch.cuda.is_available()` → True
- [ ] `from src.model import PoseSplatter` 성공
- [ ] GPU 메모리 8GB+ 확보

---

## 🔧 트러블슈팅 (1분 진단)

### Q1: `CUDA not available`?
```bash
nvidia-smi  # GPU 확인
nvcc --version  # CUDA 확인
# 둘 다 실패 → NVIDIA Driver 설치 필요
```

### Q2: `gsplat` 설치 실패?
```bash
pip install gsplat --no-cache-dir
# 여전히 실패 → `sudo apt install build-essential`
```

### Q3: `ModuleNotFoundError: No module named 'src'`?
```bash
cd /path/to/pose-splatter  # 프로젝트 루트로 이동
pwd  # 현재 디렉토리 확인
```

---

## 📚 다음 단계

### 데이터 없이 가능:
1. 코드 구조 탐색
2. Config 파일 분석 (`configs/`)
3. 모델 아키텍처 이해

### 데이터 있을 때:
1. **전처리**: `README.md` Step 1-5
2. **학습**: `python train_script.py config.json`
3. **평가**: `python evaluate_model.py config.json`

자세한 내용: `ANALYSIS_REPORT.md` 참조

---

## 💡 유용한 명령어

```bash
# 환경 활성화
conda activate pose-splatter

# GPU 모니터링
watch -n 1 nvidia-smi

# 학습 중 Loss 확인
tail -f project_directory/loss.pdf  # (PDF viewer에서)

# 환경 삭제
conda deactivate
conda env remove -n pose-splatter
```

---

**Happy Splatting!** 🎨
