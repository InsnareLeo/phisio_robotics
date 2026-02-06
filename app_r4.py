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
import cv2  # packages.txt 설치 후 정상 작동함
from unittest.mock import MagicMock

# ==========================================
# 0. 시스템 패치 (필수)
# ==========================================
# IPython 제거 (YOLOv9 호환)
sys.modules["IPython"] = MagicMock()
sys.modules["IPython.display"] = MagicMock()

# Linux(Cloud) 경로 호환
pathlib.WindowsPath = pathlib.PosixPath

# PyTorch 2.6+ 보안 에러 방지 (Weights Only 해제)
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 1. 환경 설정
# ==========================================
st.set_page_config(page_title="Phisio AI Pro (System Fixed)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 큐 설정
if 'result_queue' not in st.session_state: st.session_state.result_queue = queue.Queue(maxsize=1)
if 'img_queue' not in st.session_state: st.session_state.img_queue = queue.Queue(maxsize=1)

# 상태 변수
if 'snapshot_img' not in st.session_state: st.session_state['snapshot_img'] = None
if 'snapshot_info' not in st.session_state: st.session_state['snapshot_info'] = None
if 'side_baseline' not in st.session_state: st.session_state['side_baseline'] = None
if 'last_kps' not in st.session_state: st.session_state['last_kps'] = None
if 'load_error' not in st.session_state: st.session_state['load_error'] = None

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
        except:
            st.error("TensorFlow Import Error"); st.stop()

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
        self.method = None
        try:
            # v9 시도
            self.model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=weights_path, force_reload=True, trust_repo=True)
            self.method = "YOLOv9"
        except Exception as e1:
            try:
                # v5 Fallback 시도
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True, trust_repo=True)
                self.method = "YOLOv5"
            except Exception as e2:
                st.session_state.load_error = f"v9:{e1} / v5:{e2}"
                self.model = None

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
            img_rgb = img_arr[:, :, ::-1] # BGR -> RGB
            results = self.model(img_rgb)
            df = results.pandas().xyxy[0]
            candidates = []
            for _, row in df.iterrows():
                cx, cy = int((row['xmin']+row['xmax'])/2), int((row['ymin']+row['ymax'])/2)
                box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                candidates.append({'center': (cx, cy), 'box': box, 'conf': row['confidence']})
            
            # 중심축 기준 필터링
            valid = [c for c in candidates if abs(c['center'][0] - mid_x) < abs(l_sh[0]-r_sh[0])*0.8]
            valid.sort(key=lambda x: x['center'][1])
            
            return valid, len(valid) >= 2
        except: return [], False

# ==========================================
# 3. 유틸리티 (Drawing)
# ==========================================
def process_yolo_keypoints_original(kps):
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

def draw_overlay(img, objs):
    vis = img.copy()
    pts = [o['center'] for o in objs]
    
    # 박스 및 번호
    for i, o in enumerate(objs):
        b = o['box']
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    # 라인 연결
    if len(pts) >= 2:
        for i in range(len(pts)-1):
            cv2.line(vis, pts[i], pts[i+1], (255, 255, 0), 2)
    return vis, pts

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
except Exception as e:
    st.error(f"모델 초기화 오류: {e}"); st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # 1. 캡처용 최신 프레임 보관
    try:
        if st.session_state.img_queue.full():
            st.session_state.img_queue.get_nowait()
        st.session_state.img_queue.put(img)
    except: pass
    
    # 2. Pose 추론
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
            if st.session_state.result_queue.full():
                st.session_state.result_queue.get_nowait()
            st.session_state.result_queue.put({'kps': kps, 'action': action_text})
    except: pass
    
    return frame

# ==========================================
# 5. UI 화면
# ==========================================
col_main, col_ctrl = st.columns([1.6, 0.4])

with col_ctrl:
    st.header("⚙️ 제어")
    
    if st.button("🛠 모델 상태", use_container_width=True):
        if sp_global and sp_global.model: st.success("모델 로드 성공")
        else: st.error(f"실패: {st.session_state.load_error}")

    st.divider()

    # 캡처 및 오버레이 버튼
    if st.button("📸 Cobb 각도 (Side)", type="primary", use_container_width=True):
        if not st.session_state.img_queue.empty():
            capture_img = st.session_state.img_queue.get() # 큐에서 이미지 꺼냄
            
            res = pm_global(capture_img, verbose=False, conf=0.1)
            if res[0].keypoints is not None:
                kps = res[0].keypoints.data[0].cpu().numpy()
                objs, success = sp_global.get_spine_points(capture_img, kps)
                
                if success:
                    # 오버레이 그리기
                    vis_img, pts = draw_overlay(capture_img, objs)
                    st.session_state['snapshot_img'] = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    st.session_state['snapshot_info'] = "측정 완료"
                else:
                    st.error("스티커 인식 실패")
            else:
                st.error("사람 인식 실패")
        else:
            st.warning("웹캠 연결 확인 필요")

    # 결과 이미지 표시
    if st.session_state['snapshot_img'] is not None:
        st.image(st.session_state['snapshot_img'], caption=st.session_state.get('snapshot_info'))

    st.divider()
    status_ph = st.empty()

with col_main:
    st.subheader("🎥 실시간 모니터링")
    webrtc_streamer(
        key="pose-main",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if st.session_state.result_queue.not_empty:
        try:
            data = st.session_state.result_queue.get_nowait()
            status_ph.info(f"동작: **{data['action']}**")
        except: pass
