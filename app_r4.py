import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import os
import queue
import av
import pathlib
import sys
import tempfile
import time
import cv2  # opencv-python-headless 설치 시 사용 가능
from unittest.mock import MagicMock

# ==========================================
# 0. 시스템 패치 (필수)
# ==========================================
sys.modules["IPython"] = MagicMock()
sys.modules["IPython.display"] = MagicMock()
pathlib.WindowsPath = pathlib.PosixPath

# PyTorch 2.6+ 보안 경고 우회
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 1. 환경 설정
# ==========================================
st.set_page_config(page_title="Phisio AI Pro (Overlay Fixed)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 큐 설정: img_queue는 최신 프레임 1장만 보관 (캡처용)
if 'result_queue' not in st.session_state: st.session_state.result_queue = queue.Queue(maxsize=1)
if 'img_queue' not in st.session_state: st.session_state.img_queue = queue.Queue(maxsize=1)

# 스냅샷 결과 저장용 세션
if 'snapshot_img' not in st.session_state: st.session_state['snapshot_img'] = None
if 'snapshot_info' not in st.session_state: st.session_state['snapshot_info'] = None
if 'side_baseline' not in st.session_state: st.session_state['side_baseline'] = None

# ==========================================
# 2. 모델 로더
# ==========================================
tf = None
layers = None
models = None

def load_tf_dependencies():
    global tf, layers, models
    if tf is None:
        try:
            import tensorflow as _tf
            from tensorflow.keras import layers as _layers
            from tensorflow.keras import models as _models
            tf = _tf
            layers = _layers
            models = _models
        except: st.error("TensorFlow Import Error"); st.stop()

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
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

class StickerProcessorHybrid:
    def __init__(self, weights_path, device=DEVICE):
        self.model = None
        try:
            self.model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=weights_path, force_reload=True, trust_repo=True)
        except:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True, trust_repo=True)
            except: self.model = None

        if self.model:
            try:
                self.model.conf = 0.15
                self.model.iou = 0.45
                self.model.eval()
                self.model.to(device)
            except: pass

    def get_spine_points(self, img_arr, kps):
        if kps is None or self.model is None: return [], False
        l_sh, r_sh = kps[5][:2], kps[6][:2]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        
        try:
            # Sticker Inference
            img_rgb = img_arr[:, :, ::-1]
            results = self.model(img_rgb)
            df = results.pandas().xyxy[0]
            candidates = []
            for _, row in df.iterrows():
                cx, cy = int((row['xmin']+row['xmax'])/2), int((row['ymin']+row['ymax'])/2)
                # 시각화용 Box 좌표도 저장
                box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                candidates.append({'center': (cx, cy), 'box': box, 'conf': row['confidence']})
            
            # Filtering
            valid = [c for c in candidates if abs(c['center'][0] - mid_x) < abs(l_sh[0]-r_sh[0])*0.8]
            valid.sort(key=lambda x: x['center'][1]) # 상하 정렬
            
            return valid, len(valid) >= 2
        except: return [], False

# ==========================================
# 3. 유틸리티 (Drawing 포함)
# ==========================================
def process_yolo_keypoints_original(kps):
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

def draw_overlay(img, objs, kps):
    """이미지 위에 스티커 박스와 척추 라인을 그리는 함수"""
    vis = img.copy()
    pts = [o['center'] for o in objs]
    
    # 박스 및 번호 그리기
    for i, o in enumerate(objs):
        b = o['box']
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    # 라인 연결 (척추선)
    if len(pts) >= 2:
        for i in range(len(pts)-1):
            cv2.line(vis, pts[i], pts[i+1], (255, 255, 0), 2)
            
    return vis, pts

def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))))

@st.cache_resource
def load_all_models():
    load_tf_dependencies()
    pm = YOLO(POSE_MODEL_NAME)
    am = build_action_model((30, 51), 5)
    if os.path.exists(ACTION_WEIGHTS_PATH):
        with open(ACTION_WEIGHTS_PATH, "rb") as f: w_list = pickle.load(f)
        am.set_weights([np.array(w) for w in w_list])
    sp = StickerProcessorHybrid(STICKER_MODEL_PATH) if os.path.exists(STICKER_MODEL_PATH) else None
    return pm, am, ['Sitting (Ready)', 'Forward_Bending', 'Back_Extension', 'Side_Bending', 'Rotation'], sp

# ==========================================
# 4. WebRTC 콜백
# ==========================================
try:
    pm_global, am_global, names_global, sp_global = load_all_models()
except Exception as e: st.error(f"Error: {e}"); st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # 1. 캡처용 프레임 저장 (최신 프레임 유지)
    try:
        if st.session_state.img_queue.full():
            st.session_state.img_queue.get_nowait()
        st.session_state.img_queue.put(img)
    except: pass
    
    # 2. 실시간 Pose 추론
    res = pm_global(img, verbose=False, conf=0.1)
    kps = None
    action_text = "Wait..."
    
    if res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
        kps = res[0].keypoints.data[0].cpu().numpy()
        feat = process_yolo_keypoints_original(kps)
        input_data = np.tile(np.expand_dims(feat, axis=0), (1, 30, 1))
        pred = am_global.predict(input_data, verbose=0)
        action_text = names_global[np.argmax(pred)]
    
    try:
        if kps is not None:
            if st.session_state.result_queue.full(): st.session_state.result_queue.get_nowait()
            st.session_state.result_queue.put({'kps': kps, 'action': action_text})
    except: pass
    
    return frame

# ==========================================
# 5. UI Layout
# ==========================================
col_main, col_ctrl = st.columns([1.6, 0.4])

with col_ctrl:
    st.header("⚙️ 제어 패널")
    
    # 측정 기능 (스냅샷 & 오버레이)
    if st.button("📸 Cobb 각도 측정 (Side Baseline)", type="primary", use_container_width=True):
        if not st.session_state.img_queue.empty():
            # 1. 큐에서 이미지 꺼내기
            capture_img = st.session_state.img_queue.get()
            
            # 2. Pose & Sticker 추론
            res = pm_global(capture_img, verbose=False, conf=0.1)
            if res[0].keypoints is not None:
                kps = res[0].keypoints.data[0].cpu().numpy()
                objs, success = sp_global.get_spine_points(capture_img, kps)
                
                if success:
                    # 3. 그림 그리기 (Overlay)
                    vis_img, pts = draw_overlay(capture_img, objs, kps)
                    
                    # 4. 각도 계산 (예시)
                    v_spine = np.array(pts[0]) - np.array(pts[-1])
                    st.session_state['side_baseline'] = v_spine
                    
                    # 5. 결과 저장 (이미지 BGR -> RGB 변환)
                    st.session_state['snapshot_img'] = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    st.session_state['snapshot_info'] = "Cobb 각도: 기준선 저장됨"
                else:
                    st.error("스티커 인식 실패")
            else:
                st.error("사람 인식 실패")
        else:
            st.warning("카메라 영상이 없습니다.")

    st.markdown("---")
    # 결과 보여주기
    if st.session_state['snapshot_img'] is not None:
        st.image(st.session_state['snapshot_img'], caption=st.session_state.get('snapshot_info', '결과'))
        
    st.divider()
    st.caption("현재 동작:")
    status_ph = st.empty()


with col_main:
    st.subheader("🎥 실시간 모니터링")
    webrtc_streamer(
        key="pose-overlay",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # 실시간 상태 텍스트 업데이트
    if st.session_state.result_queue.not_empty:
        try:
            data = st.session_state.result_queue.get_nowait()
            status_ph.info(f"**{data['action']}**")
        except: pass
