# UrbanSound8K 城市聲音分類系統

## 專案結構

```
project/
├── app.py                  # Flask 後端主程式
├── requirements.txt        # Python 依賴套件
├── templates/
│   └── index.html          # 前端頁面（自動被 Flask 載入）
├── models/                 # ← 把你的模型放這裡
│   ├── lr_model.pkl        # Logistic Regression（必要）
│   ├── rf_model.pkl        # Rain Forest
│   ├── mlp_model.keras     # Keras MLP（必要）
│   ├── scaler.pkl          # StandardScaler（MLP 用，若有）
│   └── cnn_model.keras     # CNN 模型（必要）
└── test_audio/             # 放測試音檔 .wav
```

---

## 快速啟動

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 放入模型檔案

將你訓練好的模型放入 `models/` 資料夾：

| 檔名 | 對應模型 | 輸入特徵 |
|------|---------|---------|
| `lr_model.pkl` | Logistic Regression | MFCC mean/std (120-dim) |
| `rf_model.pkl` | Rain Forest | MFCC mean/std (120-dim) |
| `mlp_model.keras` | Keras MLP | 完整 MFCC 攤平 (40×174 = 6960-dim) |
| `scaler.pkl` | StandardScaler | MLP 的前處理 scaler |
| `cnn_model.keras` | CNN | Mel-spectrogram (1, 128, 128, 1) |

### 3. 啟動伺服器

```bash
python app.py
```

瀏覽器開啟 [http://localhost:5000](http://localhost:5000)

---
