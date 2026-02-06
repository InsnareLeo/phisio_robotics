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
from unittest.mock import MagicMock

# ==========================================
# 0. 시스템 설정 (IPython Mock & Path)
# ==========================================
# YOLO 모델 내부의 불필요한 의존성 제거
sys.modules["IPython"] = MagicMock()
sys.modules["IPython.display"] = MagicMock()

# Linux(Cloud)에서 Windows 경로 호환성 해결
pathlib.WindowsPath = pathlib.PosixPath

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 1. 환경 설정
# ==========================================
st.set_page_config(page_title="Phisio AI Pro (Final Fix)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=1)
if 'load_error' not in st.session_state:
    st.session_state.load_error = None

# ==========================================
# 2. TensorFlow Lazy Loading
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
# 3. 모델 클래스
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
        
        # [시도 1] YOLOv9 (WongKinYiu) 방식으로 로드 시도
        try:
            # force_reload=True로 설정하여 캐시 문제 해결 시도
            self.model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=weights_path, force_reload=True, trust_repo=True)
            self.method = "YOLOv9"
        except Exception as e_v9:
            print(f"v9 load failed: {e_v9}")
            st.session_state.load_error = f"v9 실패: {str(e_v9)}"
            
            # [시도 2] 실패 시 YOLOv5 (Ultralytics) 방식으로 재시도 (Fallback)
            # YOLOv9 모델 파일(.pt)은 대부분 YOLOv5 로더와 호환됩니다.
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True, trust_repo=True)
                self.method = "YOLOv5-Fallback"
                # 성공 시 에러 메시지 초기화
                st.session_state.load_error = None 
            except Exception as e_v5:
                # 둘 다 실패한 경우
                st.session_state.load_error = f"v9실패({str(e_v9)}) / v5실패({str(e_v5)})"
                self.model = None

        if self.model:
            try:
                self.model.conf = 0.15
                self.model.iou = 0.45
                self.model.eval()
                self.model.to(device)
            except:
                pass

    def get_spine_points(self, img_arr, kps):
        if kps is None or self.model is None: return [], False, "인식 불가"
        
        l_sh, r_sh = kps[5][:2], kps[6][:2]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        
        try:
            # 모델 입력: RGB
            img_rgb = img_arr[:, :, ::-1]
            results = self.model(img_rgb)
            
            # 결과 파싱
            df = results.pandas().xyxy[0] 
            candidates = []
            for _, row in df.iterrows():
                cx = (row['xmin'] + row['xmax']) / 2
                cy = (row['ymin'] + row['ymax']) / 2
                candidates.append({'center': (cx, cy), 'conf': row['confidence']})
            
            x_tol = abs(l_sh[0] - r_sh[0]) * 0.6
            valid_cands = [c for c in candidates if abs(c['center'][0] - mid_x) < x_tol]
            valid_cands.sort(key=lambda x: x['center'][1])
            
            if len(valid_cands) >= 2:
                return valid_cands, True, "성공"
            return valid_cands, False, f"부족 ({len(valid_cands)}개)"
            
        except Exception as e:
            return [], False, f"추론 오류: {e}"

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
    
    # Hybrid Processor 사용
    sp = StickerProcessorHybrid(STICKER_MODEL_PATH) if os.path.exists(STICKER_MODEL_PATH) else None
    
    return pm, am, ['Sitting (Ready)', 'Forward_Bending', 'Back_Extension', 'Side_Bending', 'Rotation'], sp

if 'last_kps' not in st.session_state: st.session_state['last_kps'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = "Waiting..."
if 'calc_result' not in st.session_state: st.session_state['calc_result'] = None

# ==========================================
# 5. WebRTC
# ==========================================
try:
    pm_global, am_global, names_global, sp_global = load_all_models()
except Exception as e:
    st.error(f"초기화 치명적 오류: {e}")
    st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    res = pm_global(img, verbose=False, conf=0.1)
    kps = None
    action_text = "No Pose"
    
    if res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
        kps = res[0].keypoints.data[0].cpu().numpy()
        
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
# 6. UI
# ==========================================
col_cam, col_info = st.columns([1.5, 1.0])

with col_cam:
    st.markdown("### 🎥 웹캠 스트림 (Hybrid Load)")
    webrtc_ctx = webrtc_streamer(
        key="pose-analysis-hybrid",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("📊 실시간 분석")
    status_cont = st.container()
    
    if webrtc_ctx.state.playing:
        try:
            data = st.session_state.result_queue.get(timeout=0.1)
            st.session_state['last_kps'] = data['kps']
            st.session_state['last_action'] = data['action']
        except queue.Empty:
            pass
            
    status_cont.info(f"동작 상태: **{st.session_state['last_action']}**")

    st.markdown("---")
    
    # 모델 상태 확인 로직 개선
    if st.button("🛠 모델 상태 재확인", use_container_width=True):
        if sp_global and sp_global.model:
            st.success(f"✅ 모델 로드 성공! (방식: {sp_global.method})")
            st.caption("v9 방식 실패 시 v5 방식으로 자동 전환되었습니다.")
        else:
            # 정확한 에러 메시지 출력
            err_msg = st.session_state.load_error if st.session_state.load_error else "알 수 없는 오류"
            st.error(f"❌ 모델 로드 실패 원인:\n{err_msg}")
            st.warning("팁: requirements.txt에 'thop'과 'pyyaml'이 있는지 확인하세요.")

    if st.button("📸 자세 각도 측정 (Pose 기반)", use_container_width=True):
         if st.session_state['last_kps'] is not None:
            kps = st.session_state['last_kps']
            sh_v = kps[6][:2] - kps[5][:2]
            angle = np.degrees(np.arctan2(sh_v[1], sh_v[0]))
            st.info(f"어깨 기울기: {angle:.1f}°")
         else:
             st.warning("포즈 데이터가 감지되지 않았습니다.")
