#!/usr/bin/env bash
# 3DDFA-V2: pose measurement for every method + the mesh-rotation baseline. Runs on CPU.
source "$(dirname "$0")/common.sh"

clone_repo 3DDFA_V2 https://github.com/cleardusk/3DDFA_V2.git 1b6c67601abffc1e9f248b291708aef0e43b55ae
make_env 3ddfa 3.11
pip_install "torch<2.6" "torchvision<0.21" "numpy<2" opencv-python onnx onnxruntime scipy cython \
    scikit-image matplotlib imageio pyyaml face-alignment

cd "$THIRD_PARTY/3DDFA_V2"
# The repo predates numpy 1.24, which removed np.long / np.int.
sed -i.bak 's/np\.long)/np.int64)/' bfm/bfm.py
sed -i.bak 's/dtype=np\.int)/dtype=np.intp)/' FaceBoxes/utils/nms/cpu_nms.pyx
# build.sh calls bare `python`; force the method venv onto PATH.
PATH="$(dirname "$PS_PYTHON"):$PATH" sh ./build.sh
"$PS_PYTHON" -c "import numpy, cv2, torch, face_alignment; print('3ddfa imports ok')"
echo "3ddfa ready -> $THIRD_PARTY/.venv_3ddfa/bin/python"
