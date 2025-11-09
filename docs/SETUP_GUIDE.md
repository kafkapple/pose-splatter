# Pose Splatter 환경 설정 가이드

**작성일**: 2025-11-08
**대상**: 처음 설치하는 사용자

---

## 목차
1. [시스템 요구사항](#1-시스템-요구사항)
2. [환경 설정 (Conda 사용)](#2-환경-설정-conda-사용)
3. [설치 검증](#3-설치-검증)
4. [일반적인 설치 문제 해결](#4-일반적인-설치-문제-해결)

---

## 1. 시스템 요구사항

### 1.1 하드웨어

**최소 사양**:
- GPU: NVIDIA GPU with CUDA support (8GB+ VRAM)
- RAM: 16GB+
- Storage: 50GB+ 여유 공간

**권장 사양**:
- GPU: NVIDIA RTX 3090 / A100 (24GB+ VRAM)
- RAM: 32GB+
- Storage: 100GB+ SSD

### 1.2 소프트웨어

**필수**:
- Linux (Ubuntu 20.04+ 권장) 또는 Windows with WSL2
- NVIDIA Driver (>=515.0)
- CUDA 11.8
- Conda (Anaconda 또는 Miniconda)

**확인 방법**:
```bash
# NVIDIA Driver 확인
nvidia-smi

# CUDA 확인
nvcc --version

# Conda 확인
conda --version
```

---

## 2. 환경 설정 (Conda 사용)

### 2.1 방법 A: environment.yml 사용 (권장)

```bash
# 1. Repository clone
git clone <repository-url>
cd pose-splatter

# 2. Conda 환경 생성
conda env create -f environment.yml

# 3. 환경 활성화
conda activate pose-splatter

# 4. torch-scatter 설치 (실패 시)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2.2 방법 B: 수동 설치

```bash
# 1. Conda 환경 생성
conda create -n pose-splatter python=3.10 -y

# 2. 환경 활성화
conda activate pose-splatter

# 3. PyTorch with CUDA 설치
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. gsplat 설치
pip install gsplat

# 5. torch-scatter 설치
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 6. 나머지 패키지 설치
pip install zarr h5py opencv-python torchmetrics matplotlib Pillow tqdm joblib numpy
```

### 2.3 소요 시간

- 다운로드 + 설치: 약 10-20분 (인터넷 속도에 따라)
- 디스크 공간: 약 5-8GB

---

## 3. 설치 검증

### 3.1 기본 검증

```bash
# 환경 활성화
conda activate pose-splatter

# Python 버전 확인
python --version
# 기대 출력: Python 3.10.x

# PyTorch 및 CUDA 확인
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# 기대 출력:
# PyTorch version: 2.0.0
# CUDA available: True

# GPU 정보 확인
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
# 기대 출력: GPU: NVIDIA GeForce RTX 3090 (또는 사용 중인 GPU)
```

### 3.2 패키지 검증

```bash
# 모든 주요 패키지 import 테스트
python << 'EOF'
import torch
import torchvision
import gsplat
import torch_scatter
import zarr
import h5py
import cv2
import torchmetrics
import matplotlib
import PIL
import tqdm
import joblib
print("All packages imported successfully!")
EOF
```

### 3.3 모델 검증

```bash
# 프로젝트 디렉토리에서 실행
cd /path/to/pose-splatter

# 모델 import 테스트
python -c "from src.model import PoseSplatter; print('PoseSplatter model import: OK')"

# U-Net 테스트
python -c "from src.unet_3d import Unet3D; print('Unet3D import: OK')"

# Shape carver 테스트
python -c "from src.shape_carver import ShapeCarver; print('ShapeCarver import: OK')"

# Config 테스트
python -c "from src.config_utils import Config; c = Config('configs/mouse_4.json'); print('Config loading: OK')"
```

### 3.4 간단한 테스트 실행

```bash
# U-Net forward pass 테스트
python src/unet_3d.py
# 기대 출력: Initial MSE between input and first 4 output channels = 0.xxxxxx
```

---

## 4. 일반적인 설치 문제 해결

### 4.1 CUDA 관련 문제

#### 문제: `CUDA not available`
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**해결 방법**:

1. **NVIDIA Driver 확인**:
   ```bash
   nvidia-smi
   # 오류 발생 시 드라이버 설치 필요
   ```

2. **PyTorch-CUDA 버전 확인**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   # None이면 CPU 버전 설치됨
   ```

3. **재설치**:
   ```bash
   conda remove pytorch torchvision -y
   conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
   ```

---

### 4.2 gsplat 설치 문제

#### 문제: `ERROR: Could not build wheels for gsplat`

**원인**:
- CUDA toolkit 미설치
- C++ compiler 부족
- 메모리 부족

**해결 방법**:

1. **CUDA toolkit 설치 확인**:
   ```bash
   nvcc --version
   # 없으면: sudo apt install nvidia-cuda-toolkit
   ```

2. **C++ compiler 설치** (Linux):
   ```bash
   sudo apt update
   sudo apt install build-essential
   ```

3. **캐시 삭제 후 재설치**:
   ```bash
   pip cache purge
   pip install gsplat --no-cache-dir
   ```

4. **소스에서 빌드** (최후 수단):
   ```bash
   git clone https://github.com/nerfstudio-project/gsplat.git
   cd gsplat
   pip install -e .
   ```

---

### 4.3 torch-scatter 설치 문제

#### 문제: `No matching distribution found for torch-scatter`

**해결 방법**:

1. **PyTorch 버전 확인**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. **해당 버전의 wheel 사용**:
   ```bash
   # PyTorch 2.0.0 + CUDA 11.8
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

   # 다른 버전은 https://pytorch-geometric.com/whl/ 참조
   ```

3. **Conda 사용** (대안):
   ```bash
   conda install pytorch-scatter -c pyg
   ```

---

### 4.4 import 오류

#### 문제: `ModuleNotFoundError: No module named 'src'`

**원인**: 프로젝트 디렉토리가 아닌 곳에서 실행

**해결 방법**:
```bash
# 프로젝트 루트로 이동
cd /path/to/pose-splatter

# 또는 PYTHONPATH 설정
export PYTHONPATH=/path/to/pose-splatter:$PYTHONPATH
```

---

### 4.5 메모리 부족

#### 문제: `MemoryError` 또는 시스템 멈춤

**원인**:
- 설치 중 메모리 부족
- Swap 공간 부족

**해결 방법**:

1. **Swap 증가** (Linux):
   ```bash
   # 현재 swap 확인
   free -h

   # Swap 파일 생성 (8GB)
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **설치 중 메모리 절약**:
   ```bash
   # 한 번에 하나씩 설치
   pip install --no-cache-dir gsplat
   pip install --no-cache-dir torch-scatter -f https://...
   ```

---

### 4.6 환경 충돌

#### 문제: 다른 프로젝트와 패키지 버전 충돌

**해결 방법**:

1. **별도 환경 사용** (항상 권장):
   ```bash
   conda deactivate
   conda env remove -n pose-splatter
   conda env create -f environment.yml
   ```

2. **환경 목록 확인**:
   ```bash
   conda env list
   ```

3. **올바른 환경 활성화**:
   ```bash
   conda activate pose-splatter
   which python  # conda 환경 경로 확인
   ```

---

## 5. 환경 관리

### 5.1 환경 삭제

```bash
# 환경 비활성화
conda deactivate

# 환경 삭제
conda env remove -n pose-splatter
```

### 5.2 환경 내보내기

```bash
# 현재 환경을 새 yml 파일로 저장
conda activate pose-splatter
conda env export > my-environment.yml
```

### 5.3 패키지 업데이트

```bash
conda activate pose-splatter

# 특정 패키지 업데이트
pip install --upgrade gsplat

# 모든 패키지 업데이트 (주의!)
# conda update --all
```

---

## 6. 다음 단계

환경 설정이 완료되면:

1. **데이터 준비**: `ANALYSIS_REPORT.md`의 "Phase 2: 데이터 준비" 참조
2. **전처리 실행**: 원본 비디오 → HDF5 → Zarr
3. **모델 학습**: `train_script.py` 실행

자세한 내용은 `ANALYSIS_REPORT.md` 및 `README.md`를 참조하세요.

---

## 7. 도움말

- **공식 문서**: (추가 예정)
- **GitHub Issues**: 문제 발생 시 issue 생성
- **관련 프로젝트**:
  - gsplat: https://github.com/nerfstudio-project/gsplat
  - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

---

**성공적인 설치를 기원합니다!** 🚀
