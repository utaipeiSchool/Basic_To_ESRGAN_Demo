# ESRGAN BSD100 影像超解析訓練與測試流程

本文件紀錄使用 [ESRGAN](https://github.com/xinntao/BasicSR) 搭配 BSD100 資料集進行 4 倍影像超解析的完整訓練與測試流程，適用於 Google Colab 環境。

---

## 📁 1. 掛載 Google Drive 並切換目錄
```python
from google.colab import drive
drive.mount('/content/drive')

import os
root_dir = "/content/drive/MyDrive/basic"
os.chdir(root_dir)
os.listdir()
```

---

## 🧱 2. Clone ESRGAN 專案
```bash
!rm -rf BasicSR
!git clone https://github.com/xinntao/BasicSR.git
%cd BasicSR
```

---

## 🔧 3. 安裝 PyTorch 與確認 CUDA 支援
```python
!pip install torch torchvision

import torch
print('Torch Version: ', torch.__version__)
print('CUDA Version: ', torch.version.cuda)
print('CUDNN Version: ', torch.backends.cudnn.version())
print('CUDA Available:', torch.cuda.is_available())
```

---

## 📦 4. 安裝 BasicSR 依賴套件
```python
import os
root_dir = "/content/drive/MyDrive/basic/BasicSR"
os.chdir(root_dir)
os.listdir()

!pip install -r requirements.txt
!python setup.py develop
```

---

## ⬇️ 5. 下載 ESRGAN 官方預訓練模型（可選）
```bash
!python scripts/download_pretrained_models.py ESRGAN
```

---

## 📁 6. 下載 BSD100 並分類 HR/LR 圖像
[資料來源 Figshare](https://figshare.com/articles/dataset/BSD100_Set5_Set14_Urban100/21586188)

```python
import os
import shutil

base_dir = '/content/drive/MyDrive/basic/BasicSR/datasets/BSD100/image_SRF_4'
gt_dir = '/content/drive/MyDrive/basic/BasicSR/datasets/BSD100/image_SRF_4HR'
lq_dir = '/content/drive/MyDrive/basic/BasicSR/datasets/BSD100/image_SRF_4_LR'

os.makedirs(gt_dir, exist_ok=True)
os.makedirs(lq_dir, exist_ok=True)

for fname in os.listdir(base_dir):
    src_path = os.path.join(base_dir, fname)
    if '_HR.png' in fname:
        shutil.copy(src_path, os.path.join(gt_dir, fname))
    elif '_LR.png' in fname:
        shutil.copy(src_path, os.path.join(lq_dir, fname))

print("✅ 分類完成，HR 和 LR 已分開。")
```

---

## 📝 7. 放置訓練設定檔
請將 `train_ESRGAN_BSD100_x4.yml` 放入：
```bash
options/train/ESRGAN/
```

---

## 🚀 8. 執行訓練
```bash
!python basicsr/train.py -opt options/train/ESRGAN/train_ESRGAN_BSD100_x4.yml
```

---

## 🧪 9. 執行測試
請將 `test_ESRGAN_BSD_100.yml` 放入：
```bash
options/test/ESRGAN/
```

請確認測試設定檔 `test_ESRGAN_BSD_100.yml` 已指向正確模型與測試資料集
```bash
!python basicsr/test.py -opt options/test/ESRGAN/test_ESRGAN_BSD_100.yml
```

---

## 🔍 10. 圖片比對視覺化
```python
from PIL import Image
import matplotlib.pyplot as plt
import os

lr_dir = '/content/drive/MyDrive/basic/BasicSR/datasets/BSD100/image_SRF_4_LR'
sr_dir = '/content/drive/MyDrive/basic/BasicSR/results/ESRGAN_BSD100_quick/visualization/BSD100_test'

for i in range(1, 101):
    img_id = f'{i:03d}'
    lr_path = os.path.join(lr_dir, f'img_{img_id}.png')
    sr_path = os.path.join(sr_dir, f'img_{img_id}_ESRGAN_BSD100_quick.png')

    if not os.path.exists(lr_path) or not os.path.exists(sr_path):
        print(f"❗ 圖片不存在: {lr_path} 或 {sr_path}")
        continue

    lr = Image.open(lr_path)
    sr = Image.open(sr_path)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(lr)
    plt.title(f"LR Input - img_{img_id}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sr)
    plt.title("ESRGAN Output")
    plt.axis('off')

    plt.show()
```

---
