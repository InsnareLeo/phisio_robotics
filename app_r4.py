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
from ultralytics import YOLO  # Pose 모델용 (v8)
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 0. 환경 및 경로 설정
# ==========================================
# [중요] Linux(Cloud)에서 Windows 경로로 저장된 모델 로드 시 오류 방지
pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Phisio AI Pro (Cloud/v9)", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=1)

# ==========================================
# 1. TensorFlow Lazy Loading (오류 방지)
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
# 2. 모델 클래스 정의
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

class StickerProcessorV9:
    def __init__(self, weights_path, device=DEVICE):
        # [핵심 변경] WongKinYiu/yolov9 레포지토리에서 구조를 불러와 로드
        # 'custom'을 사용하면 로컬의 weights_path(.pt)를 사용합니다.
        try:
            # force_reload=True는 캐시 문제 방지용
            self.model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=weights_path, force_reload=False, trust_repo=True)
            self.model.conf = 0.15  # Confidence Threshold
            self.model.iou = 0.45   # NMS IoU Threshold
            self.model.eval()       # 평가 모드
            self.model.to(device)
        except Exception as e:
            st.error(f"YOLOv9 모델 로드 실패: {e}\n(인터넷 연결이 필요하며, best.pt 경로가 정확해야 합니다)")
            self.model = None

    def get_spine_points(self, img_arr, kps):
        if kps is None or self.model is None: return [], False, "인식 불가"
        
        # 1. Pose 좌표 추출
        l_sh, r_sh = kps[5][:2], kps[6][:2]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        
        # 2. YOLOv9 추론
        # torch.hub 모델은 내부적으로 전처리(letterbox)를 수행하므로 RGB 이미지만 넘기면 됨
        try:
            # WebRTC 프레임(img_arr)은 BGR -> 모델 입력은 RGB
            img_rgb = img_arr[:, :, ::-1]
            results = self.model(img_rgb)
            
            # 3. 결과 파싱 (Pandas)
            df = results.pandas().xyxy[0] # xmin, ymin, xmax, ymax, confidence, class, name
            
            candidates = []
            for _, row in df.iterrows():
                cx = (row['xmin'] + row['xmax']) / 2
                cy = (row['ymin'] + row['ymax']) / 2
                candidates.append({'center': (cx, cy), 'conf': row['confidence']})
            
            # 4. 척추 중심 필터링
            x_tol = abs(l_sh[0] - r_sh[0]) * 0.6
            valid_cands = [c for c in candidates if abs(c['center'][0] - mid_x) < x_tol]
            valid_cands.sort(key=lambda x: x['center'][1]) # 위에서 아래로 정렬
            
            if len(valid_cands) >= 2:
                return valid_cands, True, "성공"
            return valid_cands, False, f"부족 ({len(valid_cands)}개)"
            
        except Exception as e:
            return [], False, f"추론 오류: {e}"

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def process_yolo_keypoints_original(kps):
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

@st.cache_resource
def load_all_models():
    # 1. TF 로드
    load_tf_dependencies()
    
    # 2. Pose Model (YOLOv8 - 공식 패키지)
    pm = YOLO(POSE_MODEL_NAME)
    
    # 3. Action Model (Keras)
    am = build_action_model((30, 51), 5)
    if os.path.exists(ACTION_WEIGHTS_PATH):
        with open(ACTION_WEIGHTS_PATH, "rb") as f: w_list = pickle.load(f)
        am.set_weights([np.array(w) for w in w_list])
    
    # 4. Sticker Model (YOLOv9 - Torch Hub)
    sp = StickerProcessorV9(STICKER_MODEL_PATH) if os.path.exists(STICKER_MODEL_PATH) else None
    
    return pm, am, ['Sitting (Ready)', 'Forward_Bending', 'Back_Extension', 'Side_Bending', 'Rotation'], sp

# 상태 변수 초기화
if 'last_kps' not in st.session_state: st.session_state['last_kps'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = "Waiting..."
if 'sticker_info' not in st.session_state: st.session_state['sticker_info'] = None

# ==========================================
# 4. WebRTC 콜백
# ==========================================
try:
    pm_global, am_global, names_global, sp_global = load_all_models()
except Exception as e:
    st.error(f"모델 초기화 중 오류: {e}")
    st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # 1. Pose 추론
    res = pm_global(img, verbose=False, conf=0.1)
    kps = None
    action_text = "No Pose"
    
    if res[0].keypoints is not None and len(res[0].keypoints.data) > 0:
        kps = res[0].keypoints.data[0].cpu().numpy()
        
        # 2. Action 추론
        feat = process_yolo_keypoints_original(kps)
        feat_tensor = np.expand_dims(feat, axis=0)
        input_data = np.tile(feat_tensor, (1, 30, 1)) 
        pred = am_global.predict(input_data, verbose=0)
        action_idx = np.argmax(pred)
        action_text = names_global[action_idx]
    
    # 3. 데이터 전송
    try:
        if kps is not None:
            if st.session_state.result_queue.full():
                st.session_state.result_queue.get_nowait()
            st.session_state.result_queue.put({'kps': kps, 'action': action_text})
    except:
        pass

    return frame

# ==========================================
# 5. UI 구성
# ==========================================
col_cam, col_info = st.columns([1.5, 1.0])

with col_cam:
    st.markdown("### 🎥 웹캠 스트림 (YOLOv9 Support)")
    webrtc_ctx = webrtc_streamer(
        key="pose-analysis-v9",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("📊 실시간 분석")
    status_cont = st.container()
    
    # 큐 데이터 수신
    if webrtc_ctx.state.playing:
        try:
            data = st.session_state.result_queue.get(timeout=0.1)
            st.session_state['last_kps'] = data['kps']
            st.session_state['last_action'] = data['action']
        except queue.Empty:
            pass
            
    status_cont.info(f"동작 상태: **{st.session_state['last_action']}**")

    st.markdown("---")
    
    # YOLOv9 테스트 버튼 (주의: 웹캠 캡처 대신 마지막 포즈 데이터만 사용 가능)
    # Cloud에서 이미지를 WebRTC 스레드 밖으로 꺼내는 것은 매우 느리므로, 
    # v9 모델 로드 성공 여부만 확인하거나 포즈 기반 계산 추천.
    
    if st.button("🛠 모델 로드 상태 확인", use_container_width=True):
        if sp_global and sp_global.model:
            st.success("✅ YOLOv9 (WongKinYiu) 모델이 정상적으로 로드되었습니다.")
            st.caption("Streamlit Cloud에서 이미지를 직접 처리하려면 이미지 업로드 기능을 사용하세요.")
        else:
            st.error("❌ 스티커 모델 로드 실패")

    if st.button("📸 자세 각도 측정 (Pose 기반)", use_container_width=True):
         if st.session_state['last_kps'] is not None:
            kps = st.session_state['last_kps']
            sh_v = kps[6][:2] - kps[5][:2]
            angle = np.degrees(np.arctan2(sh_v[1], sh_v[0]))
            st.info(f"어깨 기울기: {angle:.1f}°")
         else:
             st.warning("포즈 데이터가 감지되지 않았습니다.")
