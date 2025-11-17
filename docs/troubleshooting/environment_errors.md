# Environment Errors Troubleshooting

## NumPy Version Conflicts

### 증상

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.1 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

### 원인

- torchvision, torch_scatter 등이 NumPy 1.x로 컴파일됨
- 시스템에 NumPy 2.0+가 설치되어 호환성 문제 발생
- conda 환경 격리가 제대로 안 될 때 발생

### 해결 방법

**방법 1: 자동 수정 스크립트 (권장)**

```bash
bash scripts/utils/fix_environment.sh
```

**방법 2: 수동 수정**

```bash
# Conda 환경 활성화
conda activate splatter

# NumPy 1.x로 다운그레이드
pip install "numpy<2.0" --force-reinstall

# torchvision 재설치
pip install --upgrade --force-reinstall torchvision
```

**방법 3: 환경 재생성**

```bash
# 기존 환경 삭제
conda deactivate
conda env remove -n splatter

# 새로 생성
conda create -n splatter python=3.10 -y
conda activate splatter

# PyTorch 설치 (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# NumPy 명시적 버전 지정
pip install "numpy<2.0"

# 나머지 패키지
pip install -r requirements.txt
```

---

## torch_scatter Missing

### 증상

```
ModuleNotFoundError: No module named 'torch_scatter'
```

### 원인

- torch_scatter가 설치되지 않음
- 또는 PyTorch/CUDA 버전과 호환되지 않는 버전 설치됨

### 해결 방법

**방법 1: PyG 공식 wheel 사용 (권장)**

```bash
conda activate splatter

# PyTorch 버전 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 예: PyTorch 2.0.0, CUDA 11.8
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**방법 2: 자동 스크립트**

```bash
bash scripts/utils/fix_environment.sh
```

**방법 3: conda-forge 사용**

```bash
conda install pytorch-scatter -c pyg
```

---

## 서버 간 환경 불일치

### 증상

- 한 서버에서는 잘 되는데 다른 서버에서는 import 오류 발생
- conda 환경이 제대로 활성화되지 않음

### 원인

- 서버마다 conda 설치 경로가 다름
- `.bashrc` 또는 `.bash_profile` 설정 차이
- 시스템 레벨 Python과 충돌

### 해결 방법

**1. Conda 초기화 확인**

```bash
# ~/.bashrc에 conda init이 있는지 확인
cat ~/.bashrc | grep conda

# 없으면 추가
conda init bash
source ~/.bashrc
```

**2. 환경 활성화 확인**

```bash
# 현재 환경 확인
echo $CONDA_DEFAULT_ENV

# 제대로 활성화
conda activate splatter
which python  # /path/to/anaconda3/envs/splatter/bin/python 이어야 함
```

**3. run_training.sh 스크립트 사용**

스크립트는 자동으로 환경을 활성화합니다:

```bash
bash scripts/training/run_training.sh configs/templates/a6000_2d.json --epochs 50
```

**4. 환경 변수 명시적 설정**

```bash
# conda 경로 찾기
which conda

# 환경 활성화
eval "$(conda shell.bash hook)"
conda activate splatter

# 학습 실행
python scripts/training/train_script.py --config configs/templates/a6000_2d.json --epochs 50
```

---

## PYTHONPATH 문제

### 증상

```
ModuleNotFoundError: No module named 'src'
```

### 원인

- PYTHONPATH에 프로젝트 루트가 없음
- conda run 사용 시 환경 변수가 상속되지 않음

### 해결 방법

**방법 1: run_training.sh 사용 (권장)**

```bash
bash scripts/training/run_training.sh configs/templates/a6000_2d.json
```

**방법 2: PYTHONPATH 수동 설정**

```bash
export PYTHONPATH="/home/joon/pose-splatter:$PYTHONPATH"
python scripts/training/train_script.py --config configs/templates/a6000_2d.json
```

**방법 3: 프로젝트 루트에서 실행**

```bash
cd /home/joon/pose-splatter
PYTHONPATH=. python scripts/training/train_script.py --config configs/templates/a6000_2d.json
```

---

## 전체 진단 체크리스트

문제 발생 시 다음 순서로 확인:

```bash
# 1. Conda 환경 확인
conda env list
conda activate splatter
echo $CONDA_DEFAULT_ENV

# 2. Python 경로 확인
which python
python --version

# 3. 패키지 버전 확인
pip list | grep -E "(numpy|torch|torchvision|torch-scatter)"

# 4. Import 테스트
python -c "import numpy, torch, torchvision, torch_scatter; print('All OK')"

# 5. PYTHONPATH 확인
echo $PYTHONPATH

# 6. 프로젝트 구조 확인
ls -la scripts/training/
ls -la src/
```

**모든 체크 통과 → 정상**
**하나라도 실패 → fix_environment.sh 실행**

```bash
bash scripts/utils/fix_environment.sh
```

---

## 빠른 참조

| 문제 | 명령어 |
|------|--------|
| NumPy 버전 충돌 | `pip install "numpy<2.0" --force-reinstall` |
| torch_scatter 없음 | `bash scripts/utils/fix_environment.sh` |
| 환경 활성화 안됨 | `conda activate splatter` |
| PYTHONPATH 없음 | `bash scripts/training/run_training.sh CONFIG` |
| 전체 진단 | `bash scripts/utils/fix_environment.sh` |

---

## 예방 조치

### 새 서버 설정 시

```bash
# 1. Conda 초기화
conda init bash
source ~/.bashrc

# 2. 환경 생성
conda create -n splatter python=3.10 -y
conda activate splatter

# 3. PyTorch 설치 (버전 고정)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. NumPy 버전 고정
pip install "numpy<2.0"

# 5. torch_scatter 설치
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 6. 나머지 패키지
pip install -r requirements.txt

# 7. 환경 검증
bash scripts/utils/fix_environment.sh
```

### requirements.txt 업데이트

프로젝트 루트에 버전 고정:

```txt
numpy<2.0
torch==2.0.0
torchvision==0.15.0
torch-scatter
# ... 기타 패키지
```

---

## 문의

문제가 계속되면:
1. `bash scripts/utils/fix_environment.sh` 실행 결과 복사
2. 오류 로그 전체 복사
3. GitHub Issues에 제보
