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
│   ├── mlp_model.pkl       # Keras MLP（必要）
│   ├── scaler.pkl          # StandardScaler（MLP 用，若有）
│   └── cnn_model.pkl       # CNN 模型（必要）
└── test_audio/             # 可選：放測試音檔 .wav
```

---

## 快速啟動

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

> 若 CNN 模型為 `.h5`，額外安裝：
> ```bash
> pip install tensorflow
> ```

### 2. 放入模型檔案

將你訓練好的模型放入 `models/` 資料夾：

| 檔名 | 對應模型 | 輸入特徵 |
|------|---------|---------|
| `lr_model.pkl` | Logistic Regression | MFCC mean/std (80-dim) |
| `mlp_model.pkl` | Keras MLP | 完整 MFCC 攤平 (40×174 = 6960-dim) |
| `scaler.pkl` | StandardScaler | MLP 的前處理 scaler |
| `cnn_model.pkl` | CNN | Mel-spectrogram (1, 128, 128, 1) |

### 3. 啟動伺服器

```bash
python app.py
```

瀏覽器開啟 [http://localhost:5000](http://localhost:5000)

---

## 特徵提取規格

### LR (Baseline)
- `librosa.feature.mfcc(n_mfcc=40)`
- 對每個 MFCC 取 mean 和 std → **80-dim 向量**

### MLP (Optimized)
- `librosa.feature.mfcc(n_mfcc=40)`
- 固定 174 frames（不足補零，超出截斷）
- 展平為 40×174 = **6960-dim**
- 套用 `scaler.pkl` 做 StandardScaler

### CNN
- `librosa.feature.melspectrogram(n_mels=128)`
- 轉 dB scale，固定 128 frames
- 形狀為 **(1, 128, 128, 1)**

---

## 注意事項

1. **特徵維度要對齊訓練時的設定**  
   若你訓練時用的 `n_mfcc`、frames 數不同，請修改 `app.py` 中的 `extract_*_features()` 函式。

2. **MLP 的 scaler**  
   若訓練時沒有單獨存 scaler，而是 pipeline 的一部分，請對應調整 `extract_mlp_features()`。

3. **CNN 輸入形狀**  
   若 CNN 訓練時用 `(128, 128)` 而非 `(128, 128, 1)`，需修改 reshape。

4. **準確度數字**  
   `MODEL_ACCS` 寫在 `app.py` 的 predict 路由中，可直接修改對應到你的實際測試集準確度。
