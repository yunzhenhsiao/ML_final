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
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    return y, sr


# 在 app.py 中修改
def extract_lr_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1) # 加上這個，湊齊 120 維
    
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_max]).reshape(1, -1)
    return feat

# 在 app.py 中修改
def extract_rf_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1) # 加上這個，湊齊 120 維
    
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_max]).reshape(1, -1)
    return feat


def extract_mlp_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # 改為計算 mean, std, max (40*3 = 120維)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_max = np.max(mfcc, axis=1)
    feat = np.hstack([mfcc_mean, mfcc_std, mfcc_max]).reshape(1, -1)
    
    if 'scaler' in MODELS:
        feat = MODELS['scaler'].transform(feat) # 此時就是 120 對 120 了
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
        pred, proba = predict_with_proba(MODELS['lr'], feat, 'lr')
        results['lr'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.21,
            'label': 'Baseline',
            'description': '基礎統計特徵 — MFCC mean/std',
        }

    # ── RF ──────────────────────────────────────────────────────────────────
    if 'rf' in MODELS:
        feat = extract_rf_features(y, sr)
        pred, proba = predict_with_proba(MODELS['rf'], feat, 'rf')
        results['rf'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.38,
            'label': 'Random Forest',
            'description': '基於決策樹的集成方法',
        }

    # ── MLP ─────────────────────────────────────────────────────────────────
    if 'mlp' in MODELS:
        feat = extract_mlp_features(y, sr)
        pred, proba = predict_with_proba(MODELS['mlp'], feat, 'mlp')
        results['mlp'] = {
            'predicted': CLASSES[pred],
            'predicted_zh': CLASS_ZH[CLASSES[pred]],
            'confidence': round(float(proba[pred]), 4),
            'proba': {CLASSES[i]: round(float(p), 4) for i, p in enumerate(proba)},
            'model_accuracy': 0.93,
            'label': 'Optimized',
            'description': 'MFCC 數值特徵 + 標準化 + 深度學習',
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
