import os
import io
import pickle
import base64
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import keras
import tensorflow 

KERAS_AVAILABLE = True

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'aiff'}

# ── Class labels ───────────────────────────────────────────────────────────────
CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing',
    'dog_bark', 'drilling', 'engine_idling',
    'gun_shot', 'jackhammer', 'siren', 'street_music'
]

CLASS_ZH = {
    'air_conditioner': '冷氣機',
    'car_horn': '汽車喇叭',
    'children_playing': '兒童玩耍',
    'dog_bark': '狗吠',
    'drilling': '電鑽',
    'engine_idling': '引擎怠速',
    'gun_shot': '槍聲',
    'jackhammer': '鑿岩機',
    'siren': '警報器',
    'street_music': '街頭音樂',
}

# ── Load models at startup ─────────────────────────────────────────────────────
MODELS = {}

def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"[INFO] Created models/ directory. Please place your pkl files there.")
        return

    # Logistic Regression
    lr_path = os.path.join(model_dir, 'lr_model.pkl')
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            MODELS['lr'] = pickle.load(f)
        print('[INFO] Loaded lr_model.pkl')

    rf_path = os.path.join(model_dir, 'rf_model.pkl')
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            MODELS['rf'] = pickle.load(f)
        print('[INFO] Loaded rf_model.pkl')

    # MLP + scaler
    mlp_path = os.path.join(model_dir, 'mlp_model.pkl')
    mlp_h5  = os.path.join(model_dir, 'mlp_model.h5')
    mlp_keras = os.path.join(model_dir, 'mlp_model.keras')
    if os.path.exists(mlp_path):
        with open(mlp_path, 'rb') as f:
            MODELS['mlp'] = pickle.load(f)
        print('[INFO] Loaded mlp_model.pkl')
    elif KERAS_AVAILABLE and os.path.exists(mlp_h5):
        MODELS['mlp'] = keras.models.load_model(mlp_h5)
        print('[INFO] Loaded mlp_model.h5')
    elif KERAS_AVAILABLE and os.path.exists(mlp_keras):
        MODELS['mlp'] = keras.models.load_model(mlp_keras)
        print('[INFO] Loaded mlp_model.keras')

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            MODELS['scaler'] = pickle.load(f)
        print('[INFO] Loaded scaler.pkl')

    # CNN (.pkl or .h5/.keras)
    cnn_pkl = os.path.join(model_dir, 'cnn_model.pkl')
    cnn_h5  = os.path.join(model_dir, 'cnn_model.h5')
    cnn_keras = os.path.join(model_dir, 'cnn_model.keras')
    if os.path.exists(cnn_pkl):
        with open(cnn_pkl, 'rb') as f:
            MODELS['cnn'] = pickle.load(f)
        print('[INFO] Loaded cnn_model.pkl')
    elif KERAS_AVAILABLE and os.path.exists(cnn_h5):
        MODELS['cnn'] = keras.models.load_model(cnn_h5)
        print('[INFO] Loaded cnn_model.h5')
    elif KERAS_AVAILABLE and os.path.exists(cnn_keras):
        MODELS['cnn'] = keras.models.load_model(cnn_keras)
        print('[INFO] Loaded cnn_model.keras')

load_models()

# ── Feature extraction ─────────────────────────────────────────────────────────

def load_audio(file_bytes, sr=22050):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True, res_type='soxr_hq')
    return y, sr


# 修改 app.py 中的特徵提取部分
def extract_common_features(y, sr):
    """
    提取 120 維特徵：MFCC Mean (40) + Std (40) + Max (40)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    
    # 拼接成 120 維向量
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_max]).reshape(1, -1)
    return feat

def extract_features(y, sr):
    # 1. 提取 MFCC (40, Frames)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    print(f"Sample rate: {sr}")
    
    # 2. 為了確保與訓練代碼 extract_mfcc(file_path) 完全一致：
    # 訓練時：mfccs_mean = np.mean(mfccs.T, axis=0)
    # 我們這裡模仿一樣的動作：
    mfccs_t = mfccs.T # 轉置成 (Frames, 40)
    
    mfccs_mean = np.mean(mfccs_t, axis=0) # (40,)
    mfccs_std = np.std(mfccs_t, axis=0)   # (40,)
    
    # 計算 Delta：必須在原本的 mfccs (40, Frames) 上計算，再轉置取平均
    delta_feat = librosa.feature.delta(mfccs)
    mfccs_delta = np.mean(delta_feat.T, axis=0) # (40,)
    
    # 3. 橫向拼接：確保順序是 Mean(40) + Std(40) + Delta(40) = 120
    feat = np.hstack([mfccs_mean, mfccs_std, mfccs_delta]).reshape(1, -1)
    print(f"Mean shape: {mfccs_mean.shape}") # 應該是 (40,)
    
    # 4. 標準化
    if 'scaler' in MODELS:
        feat = MODELS['scaler'].transform(feat)
    
    return feat

def extract_lr_features(y, sr):
    # 提取基礎 MFCC (40維)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 1. 計算平均值 (Mean) - 40維
    mfcc_mean = np.mean(mfcc.T, axis=0) 
    
    # 2. 計算標準差 (Std) - 40維
    mfcc_std = np.std(mfcc.T, axis=0)
    
    # 3. 計算一階差分 (Delta) 的平均值 - 40維
    # 這是你訓練模型時真正使用的第三組特徵
    mfcc_delta = np.mean(librosa.feature.delta(mfcc).T, axis=0) 
    
    # 橫向拼接：40 + 40 + 40 = 120 維
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_delta]).reshape(1, -1)
    return feat

def extract_rf_features(y, sr):
    # 提取基礎 MFCC (40維)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 1. 計算平均值 (Mean) - 40維
    mfcc_mean = np.mean(mfcc.T, axis=0) 
    
    # 2. 計算標準差 (Std) - 40維
    mfcc_std = np.std(mfcc.T, axis=0)
    
    # 3. 計算一階差分 (Delta) 的平均值 - 40維
    # 這是確保隨機森林 (RF) 能正確分類的關鍵特徵
    mfcc_delta = np.mean(librosa.feature.delta(mfcc).T, axis=0) 
    
    # 橫向拼接：確保輸出的維度是模型預期的 (1, 120)
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_delta]).reshape(1, -1)
    return feat

def extract_mlp_features(y, sr):
    # 提取基礎 MFCC (40維)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 1. 計算平均值 (Mean) - 40維
    mfcc_mean = np.mean(mfcc.T, axis=0) 
    
    # 2. 計算標準差 (Std) - 40維
    mfcc_std = np.std(mfcc.T, axis=0)
    
    # 3. 計算一階差分 (Delta) 的平均值 - 40維
    # 必須將原本錯誤的 mfcc_max 改回訓練時用的 Delta 特徵
    mfcc_delta = np.mean(librosa.feature.delta(mfcc).T, axis=0) 
    
    # 橫向拼接成 120 維向量
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_delta]).reshape(1, -1)
    
    # 4. 使用訓練時產生的 scaler 進行標準化
    # 建議加上 'scaler' in MODELS 的判斷，防止 key 不存在時報錯
    if 'scaler' in MODELS and MODELS['scaler']:
        feat = MODELS['scaler'].transform(feat) 
        
    return feat


import cv2

def extract_cnn_features(y, sr):
    # 1. 生成 Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 2. 標準化 (請確認這段是否跟 Kaggle 一樣)
    # 如果你在 Kaggle 是縮放到 0~255，這裡也要改
    S_db_min = S_db.min()
    S_db_max = S_db.max()
    if S_db_max - S_db_min > 0:
        S_db = (S_db - S_db_min) / (S_db_max - S_db_min)
    
    # 3. 縮放與維度處理
    
    img_resized = cv2.resize(S_db, (128, 128), interpolation=cv2.INTER_AREA)
    
    # 4. 增加維度以符合 CNN 輸入 (1, 128, 128, 1)
    feat = img_resized.reshape(1, 128, 128, 1)
    return feat

import random

@app.route('/load_test', methods=['GET'])
def load_test_audio():
    filename = request.args.get('file')
    test_dir = 'test_audio'
    
    # 建立目錄確保不報錯
    if not os.path.exists(test_dir):
        return jsonify({'error': 'test_audio 資料夾不存在'}), 404
        
    if not filename:
        audio_files = [f for f in os.listdir(test_dir) if f.endswith(('.wav', '.mp3'))]
        if not audio_files: return jsonify({'error': '資料夾內無檔案'}), 404
        filename = random.choice(audio_files)
        
    filepath = os.path.join(test_dir, filename)
    
    try:
        # 核心：直接讀取原始二進位位元組，不經過任何音訊處理
        with open(filepath, 'rb') as f:
            audio_data = f.read()
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            
        return jsonify({
            'filename': filename,
            'audio_base64': encoded_audio,
            'mime_type': 'audio/wav' if filename.lower().endswith('.wav') else 'audio/mp3'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Predict helpers ────────────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_with_proba(model, feat, model_type):
    """Return (predicted_class_index, proba_array_len_10)."""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feat)[0]
        elif hasattr(model, 'predict'):
            raw = model.predict(feat)
            if raw.ndim == 2 and raw.shape[1] == 10:
                # Keras softmax output
                proba = raw[0]
            else:
                # Decision function or single class output
                if hasattr(model, 'decision_function'):
                    df = model.decision_function(feat)[0]
                    proba = softmax(df)
                else:
                    proba = np.zeros(10)
                    proba[int(raw[0])] = 1.0
        else:
            proba = np.ones(10) / 10
        pred = int(np.argmax(proba))
        return pred, proba.tolist()
    except Exception as e:
        print(f'[WARN] Prediction failed for {model_type}: {e}')
        # Return uniform distribution as fallback
        proba = np.ones(10) / 10
        return 0, proba.tolist()

# ── Visualization ──────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0d0d0d', dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def make_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(8, 2))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    times = np.linspace(0, len(y) / sr, len(y))
    ax.fill_between(times, y, alpha=0.85, color='#00e5ff')
    ax.plot(times, y, color='#00e5ff', linewidth=0.5, alpha=0.6)
    ax.set_xlabel('Time (s)', color='#888', fontsize=8)
    ax.set_ylabel('Amplitude', color='#888', fontsize=8)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222')
    ax.set_xlim(0, times[-1])
    fig.tight_layout(pad=0.5)
    return fig_to_base64(fig)


def make_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis='time', y_axis='mel',
        ax=ax, cmap='magma'
    )
    fig.colorbar(img, ax=ax, format='%+2.0f dB',
                 label='dB', orientation='vertical')
    ax.set_xlabel('Time (s)', color='#888', fontsize=8)
    ax.set_ylabel('Hz', color='#888', fontsize=8)
    ax.tick_params(colors='#555')
    for spine in ax.spines.values():
        spine.set_edgecolor('#222')
    ax.yaxis.label.set_color('#888')
    ax.xaxis.label.set_color('#888')
    fig.tight_layout(pad=0.5)
    return fig_to_base64(fig)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models_status')
def models_status():
    return jsonify({
        'lr':  'lr'  in MODELS,
        'rf':  'rf'  in MODELS,
        'mlp': 'mlp' in MODELS,
        'cnn': 'cnn' in MODELS,
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    file_bytes = file.read()

    try:
        y, sr = load_audio(file_bytes)
    except Exception as e:
        return jsonify({'error': f'Failed to load audio: {e}'}), 400

    # ── Visualizations ──────────────────────────────────────────────────────
    waveform_b64 = make_waveform(y, sr)
    mel_b64      = make_mel_spectrogram(y, sr)

    results = {}

    # ── LR ──────────────────────────────────────────────────────────────────
    if 'lr' in MODELS:
        feat = extract_lr_features(y, sr)
        raw_pred = MODELS['lr'].predict(feat) # 這是模型吐出的數字
        print(f"DEBUG - LR Raw Prediction Index: {raw_pred}")
        raw_idx = MODELS['lr'].predict(feat)[0]  # 模型預測的索引
        model_name = CLASSES[raw_idx]           # 你對應出來的名稱
        print(f"--- DEBUG ---")
        print(f"模型預測索引 (Index): {raw_idx}")
        print(f"對應類別名稱 (Name): {model_name}")
        print(f"所有類別順序 (CLASSES): {CLASSES}")
        # 注意：如果 LR 訓練時也使用了 scaler，這裡的 feat 也需要先經由 scaler 轉換
        if 'scaler' in MODELS:
            feat = MODELS['scaler'].transform(feat)
            
        pred, proba = predict_with_proba(MODELS['lr'], feat, 'lr')
        results['lr'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.60, # 根據 index.html 顯示的 59.88%
            'label': 'Baseline',
            'description': '120維 MFCC 特徵 (Mean/Std/Max) + 邏輯迴歸',
        }

    # ── RF ──────────────────────────────────────────────────────────────────
    if 'rf' in MODELS:
        feat = extract_rf_features(y, sr)
        # 注意：隨機森林通常對縮放不敏感，但若訓練時有做 normalize，建議也補上轉換
        if 'scaler' in MODELS:
            feat = MODELS['scaler'].transform(feat)

        pred, proba = predict_with_proba(MODELS['rf'], feat, 'rf')
        results['rf'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.66, # 根據 index.html 顯示的 66.29%
            'label': 'Ensemble',
            'description': '120維 MFCC 特徵 + 隨機森林集成',
        }

    # ── MLP ─────────────────────────────────────────────────────────────────
    if 'mlp' in MODELS:
        feat = extract_mlp_features(y, sr) # 內部已處理過 scaler
        pred, proba = predict_with_proba(MODELS['mlp'], feat, 'mlp')
        results['mlp'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.66, # 根據 index.html 顯示的 65.69%
            'label': 'Optimized',
            'description': '120維 MFCC 特徵 + 三層 Dense 層 (256-512-256)',
        }

    # ── CNN ─────────────────────────────────────────────────────────────────
    if 'cnn' in MODELS:
        feat = extract_cnn_features(y, sr)
        pred, proba = predict_with_proba(MODELS['cnn'], feat, 'cnn')
        results['cnn'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.72,
            'label': 'CNN',
            'description': 'Mel-spectrogram 影像識別',
        }

    duration = round(float(len(y) / sr), 2)

    return jsonify({
        'waveform': waveform_b64,
        'mel_spectrogram': mel_b64,
        'duration': duration,
        'sample_rate': sr,
        'results': results,
        'classes': CLASSES,
        'classes_zh': CLASS_ZH,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
