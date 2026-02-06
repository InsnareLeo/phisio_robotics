import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
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

# CV2 제거 및 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 실시간 데이터 공유를 위한 Queue (Thread-safe)
if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=1)

# ==========================================
# 1. 모델 및 처리 클래스
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

class StickerProcessor:
    def __init__(self, weights_path, device=DEVICE):
        # Ultralytics YOLO 네이티브 로드 (CV2 의존성 제거)
        self.model = YOLO(weights_path)
    
    def get_spine_points(self, img_arr, kps):
        # kps: Pose Keypoints
        if kps is None: return [], False, "Pose 인식 불가"
        
        # 1. Pose 기준 ROI 설정 (Numpy 연산)
        l_sh, r_sh = kps[5][:2], kps[6][:2]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        
        # 2. 스티커 모델 추론 (Ultralytics 내부 전처리 사용)
        results = self.model.predict(img_arr, verbose=False, conf=0.1)
        
        # 3. 결과 파싱 (Boxes -> Numpy)
        candidates = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
            for box in boxes:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                candidates.append({'center': (cx, cy), 'conf': box[4]})
        
        # 4. 척추 포인트 필터링 로직 (간소화됨)
        # 중심축(mid_x)에서 너무 먼 것은 제외
        x_tol = abs(l_sh[0] - r_sh[0]) * 0.5
        valid_cands = [c for c in candidates if abs(c['center'][0] - mid_x) < x_tol]
        
        # Y축 기준으로 정렬 (위 -> 아래)
        valid_cands.sort(key=lambda x: x['center'][1])
        
        if len(valid_cands) >= 2: # 최소 2개 이상이면 계산 가능으로 간주
            return valid_cands, True, "성공"
        return valid_cands, False, f"스티커 부족 ({len(valid_cands)}개)"

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))))

def process_yolo_keypoints_original(kps):
    # Numpy 연산만 사용
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

@st.cache_resource
def load_all_models():
    # Pose Model
    pm = YOLO(POSE_MODEL_NAME)
    
    # Action Model
    am = build_action_model((30, 51), 5)
    if os.path.exists(ACTION_WEIGHTS_PATH):
        with open(ACTION_WEIGHTS_PATH, "rb") as f: w_list = pickle.load(f)
        am.set_weights([np.array(w) for w in w_list])
    
    # Sticker Model
    sp = StickerProcessor(STICKER_MODEL_PATH) if os.path.exists(STICKER_MODEL_PATH) else None
    
    return pm, am, ['Sitting (Ready)', 'Forward_Bending', 'Back_Extension', 'Side_Bending', 'Rotation'], sp

# 세션 초기화
if 'last_kps' not in st.session_state: st.session_state['last_kps'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = "Waiting..."
for k in ['side_baseline_vec', 'rot_baseline_vec', 'error_msg', 'calc_result']:
    if k not in st.session_state: st.session_state[k] = None

# ==========================================
# 3. WebRTC 콜백 (별도 스레드)
# ==========================================
# 전역 변수로 모델 로드 (스레드 접근용)
pm_global, am_global, names_global, sp_global = load_all_models()

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
        feat_tensor = np.expand_dims(feat, axis=0) # (1, 51)
        # Action 모델 입력 차원 (Batch, Time, Features) -> (1, 30, 51) 필요
        # 단일 프레임 처리 로직: 30프레임 버퍼가 없으면 복제
        input_data = np.tile(feat_tensor, (1, 30, 1)) 
        
        pred = am_global.predict(input_data, verbose=0)
        action_idx = np.argmax(pred)
        action_text = names_global[action_idx]
        
    # 3. 메인 스레드로 데이터 전송 (이미지 그리기 없음)
    try:
        if kps is not None:
            # 큐에 최신 데이터 덮어쓰기
            if st.session_state.result_queue.full():
                st.session_state.result_queue.get_nowait()
            st.session_state.result_queue.put({'kps': kps, 'action': action_text})
    except:
        pass

    return frame

# ==========================================
# 4. 메인 UI
# ==========================================
col_cam, col_info = st.columns([1.5, 1.0])

with col_cam:
    st.markdown("### 🎥 웹캠 스트림 (WebRTC)")
    # WebRTC 스트리머 설정
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
    
    # 실시간 정보 표시용 컨테이너
    status_container = st.container()
    result_container = st.empty()
    
    # 큐에서 데이터 폴링 및 세션 업데이트
    if webrtc_ctx.state.playing:
        try:
            data = st.session_state.result_queue.get(timeout=0.1)
            st.session_state['last_kps'] = data['kps']
            st.session_state['last_action'] = data['action']
        except queue.Empty:
            pass
            
    status_container.info(f"현재 동작: **{st.session_state['last_action']}**")

    st.markdown("---")
    st.subheader("🛠️ 측정 도구")

    # Cobb 각도 (데이터 기반 계산)
    if st.button("📸 Cobb 각도 측정 (Side Baseline)", type="primary", use_container_width=True):
        if st.session_state['last_kps'] is not None and sp_global:
            # 현재 프레임 이미지는 없으므로 더미 이미지 생성하여 위치만 파악 (또는 로직 분리)
            # 여기서는 편의상 KPS와 Sticker 위치 관계만 계산해야 하지만, 
            # Sticker 모델은 이미지가 필요함.
            # *제약사항*: WebRTC 콜백 밖에서는 이미지를 얻기 어려움.
            # 따라서 'snapshot' 대신 '마지막 인식된 상태'를 텍스트로 안내
            st.warning("⚠️ Cloud 모드: 이미지 캡처 대신 실시간 데이터를 사용합니다.")
            
            # (Note: 실제 이미지가 없으면 Sticker 모델을 돌릴 수 없습니다. 
            #  Cloud WebRTC 구조상 이미지를 메인 스레드로 가져오는 것은 대역폭 문제가 있습니다.
            #  따라서 이 기능은 'Pose Keypoint' 기반의 간단한 각도로 대체하거나 
            #  기능 제한 메시지를 띄우는 것이 안전합니다.)
            
            # 여기서는 동작 확인을 위해 Pose 데이터로 대체 계산 예시
            kps = st.session_state['last_kps']
            sh_vector = kps[6][:2] - kps[5][:2] # 어깨 기울기
            hip_vector = kps[12][:2] - kps[11][:2] # 골반 기울기
            # 간단한 척추 정렬 각도 (대체)
            angle = angle_between(sh_vector, hip_vector)
            st.session_state['calc_result'] = f"상체-하체 정렬 각도: {angle:.1f}°"
        else:
            st.error("데이터가 없습니다. 웹캠이 켜져 있나요?")

    if st.session_state['calc_result']:
        st.success(st.session_state['calc_result'])

    st.markdown("---")
    st.caption("※ Streamlit Cloud 환경에서는 cv2 그래픽 처리가 제한되어 텍스트 결과만 제공됩니다.")
