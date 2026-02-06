import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import os
import queue
import av
from collections import deque
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 0. 환경 및 경로 설정
# ==========================================
st.set_page_config(page_title="Phisio AI Pro (Cloud)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=1)

# ==========================================
# 1. 모델 및 처리 클래스 (Lazy Loading 적용)
# ==========================================
# 전역 변수로 선언하여 함수 내에서 할당
tf = None
layers = None
models = None

def load_tf_dependencies():
    """TensorFlow 의존성을 필요할 때 로드"""
    global tf, layers, models
    if tf is None:
        try:
            import tensorflow as _tf
            from tensorflow.keras import layers as _layers
            from tensorflow.keras import models as _models
            tf = _tf
            layers = _layers
            models = _models
        except ImportError as e:
            st.error(f"TensorFlow 로드 실패: {e}")
            st.stop()
        except Exception as e:
            st.error(f"시스템 오류: {e}")
            st.stop()

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # layers가 로드된 상태라고 가정
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_action_model(input_shape, n_classes):
    # layers, models 사용
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    for _ in range(2): 
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

class StickerProcessor:
    def __init__(self, weights_path, device=DEVICE):
        self.model = YOLO(weights_path)
    
    def get_spine_points(self, img_arr, kps):
        if kps is None: return [], False, "Pose 인식 불가"
        l_sh, r_sh = kps[5][:2], kps[6][:2]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        results = self.model.predict(img_arr, verbose=False, conf=0.1)
        candidates = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()
            for box in boxes:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                candidates.append({'center': (cx, cy), 'conf': box[4]})
        
        x_tol = abs(l_sh[0] - r_sh[0]) * 0.5
        valid_cands = [c for c in candidates if abs(c['center'][0] - mid_x) < x_tol]
        valid_cands.sort(key=lambda x: x['center'][1])
        
        if len(valid_cands) >= 2: return valid_cands, True, "성공"
        return valid_cands, False, f"스티커 부족 ({len(valid_cands)}개)"

# ==========================================
# 2. 유틸리티 및 모델 로더
# ==========================================
def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))))

def process_yolo_keypoints_original(kps):
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

@st.cache_resource
def load_all_models():
    # 1. TensorFlow 지연 로드
    load_tf_dependencies()
    
    # 2. 모델 빌드
    pm = YOLO(POSE_MODEL_NAME)
    am = build_action_model((30, 51), 5)
    
    if os.path.exists(ACTION_WEIGHTS_PATH):
        with open(ACTION_WEIGHTS_PATH, "rb") as f: w_list = pickle.load(f)
        am.set_weights([np.array(w) for w in w_list])
    
    sp = StickerProcessor(STICKER_MODEL_PATH) if os.path.exists(STICKER_MODEL_PATH) else None
    
    return pm, am, ['Sitting (Ready)', 'Forward_Bending', 'Back_Extension', 'Side_Bending', 'Rotation'], sp

# 세션 및 상태 초기화
if 'last_kps' not in st.session_state: st.session_state['last_kps'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = "Waiting..."
for k in ['calc_result']:
    if k not in st.session_state: st.session_state[k] = None

# ==========================================
# 3. WebRTC 로직
# ==========================================
# 모델 전역 로드 (캐싱됨)
try:
    pm_global, am_global, names_global, sp_global = load_all_models()
except Exception as e:
    st.error(f"모델 로딩 중 치명적 오류 발생: {e}")
    st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Pose 추론
    res = pm_global(img, verbose=False, conf=0.1)
    kps = None
    action_text = "No Pose"
    
    if res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
        kps = res[0].keypoints.data[0].cpu().numpy()
        
        # Action 추론 (TensorFlow)
        feat = process_yolo_keypoints_original(kps)
        feat_tensor = np.expand_dims(feat, axis=0)
        input_data = np.tile(feat_tensor, (1, 30, 1)) 
        
        pred = am_global.predict(input_data, verbose=0)
        action_idx = np.argmax(pred)
        action_text = names_global[action_idx]
        
    try:
        if kps is not None:
            if st.session_state.result_queue.full():
                st.session_state.result_queue.get_nowait()
            st.session_state.result_queue.put({'kps': kps, 'action': action_text})
    except:
        pass

    return frame

# ==========================================
# 4. UI 구성
# ==========================================
col_cam, col_info = st.columns([1.5, 1.0])

with col_cam:
    st.markdown("### 🎥 웹캠 스트림 (WebRTC)")
    webrtc_ctx = webrtc_streamer(
        key="pose-analysis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("📊 실시간 분석 결과")
    status_container = st.container()
    
    if webrtc_ctx.state.playing:
        try:
            data = st.session_state.result_queue.get(timeout=0.1)
            st.session_state['last_kps'] = data['kps']
            st.session_state['last_action'] = data['action']
        except queue.Empty:
            pass
            
    status_container.info(f"현재 동작: **{st.session_state['last_action']}**")

    st.markdown("---")
    if st.button("📸 자세 측정 (Snap)", type="primary", use_container_width=True):
        if st.session_state['last_kps'] is not None:
            kps = st.session_state['last_kps']
            sh_vector = kps[6][:2] - kps[5][:2]
            hip_vector = kps[12][:2] - kps[11][:2]
            angle = angle_between(sh_vector, hip_vector)
            st.session_state['calc_result'] = f"어깨-골반 정렬: {angle:.1f}°"
        else:
            st.error("데이터 없음")

    if st.session_state['calc_result']:
        st.success(st.session_state['calc_result'])
