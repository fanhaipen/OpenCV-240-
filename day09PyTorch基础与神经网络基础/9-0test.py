# test_all_installations.py
import torch
import sklearn
import matplotlib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import seaborn as sns
import jupyter

print("✅ 所有库安装验证")
print("=" * 50)

libraries = {
    "PyTorch": torch.__version__,
    "scikit-learn": sklearn.__version__,
    "Matplotlib": matplotlib.__version__,
    "NumPy": np.__version__,
    "Pandas": pd.__version__,
    "OpenCV": cv2.__version__,
    "PIL (Pillow)": Image.__version__,
    "Seaborn": sns.__version__,
}

for lib, version in libraries.items():
    print(f"{lib:15} : {version}")

print(f"{'CUDA可用':15} : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"{'CUDA版本':15} : {torch.version.cuda}")
    print(f"{'GPU数量':15} : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"{'GPU名称':15} : {torch.cuda.get_device_name(i)}")

print("=" * 50)
print("✅ 所有库安装成功！")