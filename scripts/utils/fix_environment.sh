#!/bin/bash

echo "================================"
echo "Environment Diagnosis & Fix"
echo "================================"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate splatter

echo ""
echo "1. Python & Conda Info:"
which python
python --version
echo "Conda env: $CONDA_DEFAULT_ENV"

echo ""
echo "2. Current Package Versions:"
pip list | grep -E "(numpy|torch|torchvision|torch-scatter)"

echo ""
echo "3. Testing imports:"
python -c "
import sys
print(f'Python: {sys.executable}')

try:
    import numpy
    print(f'✓ NumPy {numpy.__version__} - {numpy.__file__}')
except Exception as e:
    print(f'✗ NumPy failed: {e}')

try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
except Exception as e:
    print(f'✗ PyTorch failed: {e}')

try:
    import torchvision
    print(f'✓ torchvision {torchvision.__version__}')
except Exception as e:
    print(f'✗ torchvision failed: {e}')

try:
    import torch_scatter
    print(f'✓ torch_scatter {torch_scatter.__version__}')
except Exception as e:
    print(f'✗ torch_scatter failed: {e}')
"

echo ""
echo "================================"
echo "Fixing Environment..."
echo "================================"

# Fix 1: Ensure NumPy < 2.0
echo ""
echo "1. Fixing NumPy version..."
pip install "numpy<2.0" --force-reinstall --no-cache-dir

# Fix 2: Install torch_scatter if missing
echo ""
echo "2. Checking torch_scatter..."
python -c "import torch_scatter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing torch_scatter..."

    # Get PyTorch and CUDA versions
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

    echo "PyTorch: $TORCH_VERSION, CUDA: $CUDA_VERSION"

    # Install from wheel (faster and more reliable)
    pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION//.}.html
else
    echo "torch_scatter already installed"
fi

# Fix 3: Reinstall torchvision with correct NumPy
echo ""
echo "3. Reinstalling torchvision..."
pip install --upgrade --force-reinstall --no-cache-dir torchvision

echo ""
echo "================================"
echo "Verification"
echo "================================"

python -c "
import numpy
import torch
import torchvision
import torch_scatter

print(f'✓ NumPy {numpy.__version__}')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ torchvision {torchvision.__version__}')
print(f'✓ torch_scatter {torch_scatter.__version__}')
print('')
print('All packages installed successfully!')
"

echo ""
echo "================================"
echo "Fix Complete!"
echo "================================"
