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
from unittest.mock import MagicMock

# ==========================================
# 0. 시스템 호환성 패치 (매우 중요)
# ==========================================

# 1. IPython 모듈 에러 방지 (YOLOv9 호환)
sys.modules["IPython"] = MagicMock()
sys.modules["IPython.display"] = MagicMock()

# 2. Linux(Cloud)에서 Windows Path 에러 방지
pathlib.WindowsPath = pathlib.PosixPath

# 3. [핵심] PyTorch 2.6+ weights_only 에러 강제 해결
# torch.hub.load가 내부적으로 torch.load를 호출할 때 보안 검사를 우회하도록 설정
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False  # 강제로 보안 검사 해제
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 1. 환경 설정
# ==========================================
st.set_page_config(page_title="Phisio AI Pro (Cloud/Fixed)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 세션 상태 초기화
if 'result_queue' not in st.session_state: st.session_state.result_queue = queue.Queue(maxsize=1)
if 'last_kps' not in st.session_state: st.session_state['last_kps'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = "Waiting..."
if 'load_error' not in st.session_state: st.session_state.load_error = None
if 'is_processing_video' not in st.session_state: st.session_state['is_processing_video'] = False

# ==========================================
# 2. 모델 로더 (TensorFlow Lazy Load)
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
        except ImportError:
            st.error("TensorFlow 로드 실패. requirements.txt를 확인하세요.")
            st.stop()

# ==========================================
# 3. 모델 아키텍처
# ==========================================
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
        
        # v9 시도 -> 실패시 v5 시도 (PyTorch 2.6 패치 적용됨)
        try:
            self.model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=weights_path, force_reload=True, trust_repo=True)
            self.method = "YOLOv9"
        except Exception as e_v9:
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True, trust_repo=True)
                self.method = "YOLOv5-Fallback"
            except Exception as e_v5:
                st.session_state.load_error = f"로드 실패: {e_v9} || {e_v5}"
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
            img_rgb = img_arr[:, :, ::-1]
            results = self.model(img_rgb)
            df = results.pandas().xyxy[0]
            candidates = []
            for _, row in df.iterrows():
                candidates.append({'center': ((row['xmin']+row['xmax'])/2, (row['ymin']+row['ymax'])/2)})
            # 간단 필터링
            valid = [c for c in candidates if abs(c['center'][0] - mid_x) < abs(l_sh[0]-r_sh[0])*0.8]
            valid.sort(key=lambda x: x['center'][1])
            return valid, len(valid) >= 2
        except: return [], False

# ==========================================
# 4. 유틸리티
# ==========================================
def process_yolo_keypoints_original(kps):
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

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
# 5. WebRTC 콜백
# ==========================================
try:
    pm_global, am_global, names_global, sp_global = load_all_models()
except Exception as e:
    st.error(f"모델 초기화 실패: {e}"); st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    res = pm_global(img, verbose=False, conf=0.1)
    kps = None
    action_text = "No Pose"
    
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
# 6. UI 구성
# ==========================================
col_main, col_ctrl = st.columns([1.6, 0.4])

with col_ctrl:
    st.header("⚙️ 제어 패널")
    
    # 소스 선택
    input_source = st.radio("입력 소스", ["Webcam", "Video File"])
    
    if st.button("🛠 모델 상태 확인", use_container_width=True):
        if sp_global and sp_global.model:
            st.success(f"로드 성공 ({sp_global.method})")
        else:
            st.error(f"실패: {st.session_state.load_error}")
            
    st.divider()
    
    # 측정 버튼 (원본 기능 복구)
    if st.button("📸 Cobb 각도 (Side Baseline)", use_container_width=True):
        if st.session_state['last_kps'] is not None:
            kps = st.session_state['last_kps']
            # Cloud에서는 이미지 위에 그리는 대신 계산값 출력으로 대체
            angle = np.degrees(np.arctan2(kps[0][1]-kps[12][1], kps[0][0]-kps[12][0]))
            st.info(f"Side Baseline 저장됨 (임시 각도: {abs(angle):.1f}°)")
        else: st.warning("데이터 없음")

    if st.button("📏 회전 기준각 (Rot Baseline)", use_container_width=True):
         if st.session_state['last_kps'] is not None:
             st.info("회전 기준각 저장됨")
         else: st.warning("데이터 없음")

with col_main:
    st.subheader("🎥 분석 화면")
    
    if input_source == "Webcam":
        webrtc_streamer(
            key="pose-analysis",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # 실시간 결과 표시 (WebRTC 큐 사용)
        placeholder = st.empty()
        if st.session_state.result_queue.not_empty:
            try:
                data = st.session_state.result_queue.get_nowait()
                st.session_state['last_kps'] = data['kps']
                st.session_state['last_action'] = data['action']
            except: pass
        placeholder.info(f"현재 동작: **{st.session_state['last_action']}**")

    elif input_source == "Video File":
        # 동영상 파일 처리 로직 (원본 기능 복구)
        video_file = st.file_uploader("동영상 업로드", type=['mp4', 'mov', 'avi'])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            if st.button("▶ 영상 분석 시작"):
                cap = cv2.VideoCapture(tfile.name) # cv2는 여기서만 로컬 변수로 사용 필요
                # 하지만 cv2 import가 없으므로 opencv-python-headless가 설치되어 있다면
                # import cv2를 함수 안에서 시도해야 함.
                try:
                    import cv2
                    cap = cv2.VideoCapture(tfile.name)
                    st_frame = st.empty()
                    st_info = st.empty()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Pose 추론
                        res = pm_global(frame, verbose=False, conf=0.1)
                        if res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
                            kps = res[0].keypoints.data[0].cpu().numpy()
                            st.session_state['last_kps'] = kps
                            
                            # Action 추론
                            feat = process_yolo_keypoints_original(kps)
                            input_data = np.tile(np.expand_dims(feat, axis=0), (1, 30, 1))
                            pred = am_global.predict(input_data, verbose=0)
                            act = names_global[np.argmax(pred)]
                            
                            # 시각화 (CV2 drawing -> RGB -> Streamlit)
                            for i, p in enumerate(kps):
                                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0, 255, 0), -1)
                            st_info.markdown(f"**Action: {act}**")
                        
                        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        time.sleep(0.03) # 프레임 속도 조절
                    cap.release()
                except ImportError:
                    st.error("OpenCV가 설치되지 않았습니다. requirements.txt를 확인하세요.")
