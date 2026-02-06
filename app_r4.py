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
import cv2
from unittest.mock import MagicMock

# ==========================================
# 0. 시스템 호환성 패치
# ==========================================
sys.modules["IPython"] = MagicMock()
sys.modules["IPython.display"] = MagicMock()
pathlib.WindowsPath = pathlib.PosixPath

_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==========================================
# 1. 환경 및 상태 설정
# ==========================================
st.set_page_config(page_title="Phisio AI Pro (Webcam Final)", layout="wide")
st.markdown("""<style>.stImage > img { width: 100%; border-radius: 8px; }</style>""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 세션 상태 초기화
keys = ['result_queue', 'img_queue', 'snapshot_result', 'side_baseline_vec', 
        'rot_baseline_vec', 'error_msg', 'rot_base_angle', 'last_frame_data']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = queue.Queue(maxsize=1) if 'queue' in k else None

# ==========================================
# 2. 모델 및 처리 클래스
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
        except: st.error("TF Load Error"); st.stop()

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

    def _get_raw_candidates(self, img_arr):
        if self.model is None: return []
        try:
            img_rgb = img_arr[:, :, ::-1]
            results = self.model(img_rgb)
            df = results.pandas().xyxy[0]
            candidates = []
            for _, row in df.iterrows():
                cx, cy = int((row['xmin']+row['xmax'])/2), int((row['ymin']+row['ymax'])/2)
                box = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
                candidates.append({'center': (cx, cy), 'box': box, 'conf': row['confidence']})
            return candidates
        except: return []

    def get_spine_points(self, img_arr, kps):
        if kps is None: return [], False, "Pose 인식 불가"
        candidates = self._get_raw_candidates(img_arr)
        if not candidates: return [], False, "스티커 미검출"

        l_sh, r_sh = kps[5][:2], kps[6][:2]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        x_tol = abs(l_sh[0] - r_sh[0]) * 0.8 
        valid_cands = [c for c in candidates if abs(c['center'][0] - mid_x) < x_tol]
        valid_cands.sort(key=lambda x: x['center'][1])
        
        if len(valid_cands) >= 2:
            return valid_cands, True, "성공"
        return valid_cands, False, f"부족 ({len(valid_cands)}개)"

    def _get_nms_candidates(self, img_bgr, roi):
        return self._get_raw_candidates(img_bgr)

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))))

def draw_spine_and_boxes(vis, objs):
    pts = [o['center'] for o in objs]
    for i, o in enumerate(objs):
        b = o['box']
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.putText(vis, str(i+1), (pts[i][0]+15, pts[i][1]), 0, 1.0, (0,0,255), 2)
    
    if len(pts) >= 6:
        cv2.line(vis, pts[0], pts[1], (255,255,0), 2); cv2.line(vis, pts[1], pts[2], (255,255,0), 2)
        cv2.line(vis, pts[3], pts[4], (255,0,255), 2); cv2.line(vis, pts[4], pts[5], (255,0,255), 2)
    elif len(pts) > 1:
         for i in range(len(pts)-1):
             cv2.line(vis, pts[i], pts[i+1], (255,255,0), 2)
    return vis, pts

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
# 4. WebRTC 콜백
# ==========================================
try:
    pm_global, am_global, names_global, sp_global = load_all_models()
except Exception as e: st.error(f"Error: {e}"); st.stop()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # 캡처용 큐
    try:
        if st.session_state.img_queue.full():
            st.session_state.img_queue.get_nowait()
        st.session_state.img_queue.put(img)
    except: pass
    
    # 실시간 분석
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
# 5. 메인 UI
# ==========================================
col_v, col_r, col_c = st.columns([1.5, 1.1, 0.9])

with col_v:
    st.markdown("### 🎥 실시간 분석")
    webrtc_streamer(
        key="pose-main",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    status_ph = st.empty()
    if st.session_state.result_queue.not_empty:
        try:
            data = st.session_state.result_queue.get_nowait()
            status_ph.info(f"현재 동작: **{data['action']}**")
        except: pass

with col_r:
    st.markdown("### 📊 측정 결과")
    r_spot = st.empty()
    
    if st.session_state['error_msg']: 
        r_spot.error(st.session_state['error_msg'])
    elif st.session_state['snapshot_result']:
        img, v1, v2, label = st.session_state['snapshot_result']
        r_spot.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        c1, c2 = st.columns(2)
        c1.metric(f"{label} (Main)", f"{v1:.1f}°")
        if v2 != 0: c2.metric(f"{label} (Sub)", f"{v2:.1f}°")
    else: 
        r_spot.info("분석 중 측정 버튼을 눌러주세요.")

with col_c:
    st.subheader("🛠️ 제어 패널")
    
    # 캡처 헬퍼 함수
    def capture_current_frame():
        if st.session_state.img_queue.empty(): return None, None
        frame = st.session_state.img_queue.get()
        res = pm_global(frame, verbose=False, conf=0.1)
        kps = res[0].keypoints.data[0].cpu().numpy() if res[0].keypoints else None
        return frame, kps

    if st.button("📸 Cobb 각도 측정 (Side Baseline 저장)", type="primary", use_container_width=True):
        st.session_state['error_msg'] = None
        f, k = capture_current_frame()
        
        if f is not None and k is not None:
            objs, success, msg = sp_global.get_spine_points(f, k)
            if success:
                vis = f.copy(); vis, pts = draw_spine_and_boxes(vis, objs)
                if len(pts) >= 6:
                    st.session_state['side_baseline_vec'] = np.array(pts[0]) - np.array(pts[5])
                    cv2.line(vis, pts[5], pts[0], (0,255,255), 6) 
                    u = angle_between(np.array(pts[0])-np.array(pts[1]), np.array(pts[2])-np.array(pts[1]))
                    l = angle_between(np.array(pts[3])-np.array(pts[4]), np.array(pts[5])-np.array(pts[4]))
                    st.session_state['snapshot_result'] = (vis, u, l, "Cobb 각도")
                else:
                    st.session_state['side_baseline_vec'] = np.array(pts[0]) - np.array(pts[-1])
                    cv2.line(vis, pts[-1], pts[0], (0,255,255), 6)
                    u = angle_between(np.array(pts[0])-np.array(pts[1]), np.array(pts[-2])-np.array(pts[-1]))
                    st.session_state['snapshot_result'] = (vis, u, 0, "Cobb 각도 (약식)")
                st.rerun()
            else: st.session_state['error_msg'] = f"측정 실패: {msg}"; st.rerun()
        else: st.session_state['error_msg'] = "영상 신호 없음"; st.rerun()

    if st.button("📏 회전 기준각 저장 (Rotation Baseline)", use_container_width=True):
        st.session_state['error_msg'] = None
        f, k = capture_current_frame()
        
        if f is not None and k is not None:
            objs, success, msg = sp_global.get_spine_points(f, k)
            if success:
                vis = f.copy(); pts = [o['center'] for o in objs]
                p6_idx = 5 if len(pts) > 5 else len(pts)-1
                p6 = np.array(pts[p6_idx])
                
                sh_l, sh_r = k[5][:2], k[6][:2]
                cands = sp_global._get_nms_candidates(f, (0, 0, f.shape[1], f.shape[0]))
                spine_pts = [tuple(p) for p in pts]
                lateral_cands = [c for c in cands if tuple(c['center']) not in spine_pts]
                
                if lateral_cands:
                    target_pt = np.array(min(lateral_cands, key=lambda c: min(np.linalg.norm(np.array(c['center'])-sh_l), np.linalg.norm(np.array(c['center'])-sh_r)))['center'])
                else:
                    target_pt = np.array(sh_l if k[5][2] > k[6][2] else sh_r)
                
                v_spine = np.array(pts[0]) - p6
                st.session_state['rot_baseline_vec'] = v_spine
                ang = angle_between(v_spine, target_pt - p6)
                
                cv2.line(vis, tuple(p6), tuple(pts[0]), (0,255,255), 6)
                cv2.line(vis, tuple(p6), tuple(target_pt.astype(int)), (255,0,0), 6)
                st.session_state['snapshot_result'] = (vis, ang, 0, "회전 기준각"); st.rerun()
            else: st.session_state['error_msg'] = f"저장 실패: {msg}"; st.rerun()
        else: st.session_state['error_msg'] = "영상 신호 없음"; st.rerun()

    st.markdown("---")

    # [수정됨] 파일 로드 -> 실시간 캡처로 변경
    if st.button("📐 측면 굴곡 측정 (실시간)", use_container_width=True):
        st.session_state['error_msg'] = None
        if st.session_state['side_baseline_vec'] is not None:
            f, k = capture_current_frame() # 웹캠 캡처
            
            if f is not None and k is not None:
                objs, success, msg = sp_global.get_spine_points(f, k)
                if success:
                    vis = f.copy(); vis, pts = draw_spine_and_boxes(vis, objs)
                    p1 = np.array(pts[0])
                    p6_idx = 5 if len(pts) > 5 else len(pts)-1
                    p6 = np.array(pts[p6_idx])
                    
                    b_vec = st.session_state['side_baseline_vec']
                    curr_vec = p1 - p6
                    scale = np.linalg.norm(curr_vec) / np.linalg.norm(b_vec)
                    
                    cv2.line(vis, tuple(p6.astype(int)), tuple((p6 + b_vec * scale).astype(int)), (0,255,255), 6)
                    cv2.line(vis, tuple(p6.astype(int)), tuple(p1.astype(int)), (255,0,0), 6)
                    st.session_state['snapshot_result'] = (vis, angle_between(b_vec, curr_vec), 0, "측면 굴곡"); st.rerun()
                else: st.session_state['error_msg'] = f"스티커 인식 실패: {msg}"; st.rerun()
            else: st.session_state['error_msg'] = "영상 신호 없음"; st.rerun()
        else: st.session_state['error_msg'] = "먼저 Baseline을 설정하세요."; st.rerun()

    # [수정됨] 파일 로드 -> 실시간 캡처로 변경
    if st.button("📐 회전 측정 (실시간)", use_container_width=True):
        st.session_state['error_msg'] = None
        if st.session_state['rot_baseline_vec'] is not None:
            f, k = capture_current_frame() # 웹캠 캡처
            
            if f is not None and k is not None:
                objs, success, msg = sp_global.get_spine_points(f, k)
                
                # 회전은 Pose만 있어도 대략 계산 가능
                p6_point = objs[5]['center'] if success and len(objs)>5 else [(k[11][0]+k[12][0])/2, (k[11][1]+k[12][1])/2]
                p6 = np.array(p6_point)
                sh_l, sh_r = k[5][:2], k[6][:2]
                
                cands = sp_global._get_nms_candidates(f, (0, 0, f.shape[1], f.shape[0]))
                spine_centers = [tuple(o['center']) for o in objs] if success else []
                lateral_cands = [c for c in cands if tuple(c['center']) not in spine_centers]
                
                if lateral_cands:
                    target = np.array(min(lateral_cands, key=lambda c: min(np.linalg.norm(np.array(c['center'])-sh_l), np.linalg.norm(np.array(c['center'])-sh_r)))['center'])
                else: target = np.array(sh_l if k[5][2] > k[6][2] else sh_r)
                
                b_vec = st.session_state['rot_baseline_vec']
                curr_vec = target - p6
                scale = np.linalg.norm(curr_vec) / np.linalg.norm(b_vec)
                
                vis = f.copy()
                cv2.circle(vis, tuple(target.astype(int)), 15, (0, 0, 255), -1)
                cv2.line(vis, tuple(p6.astype(int)), tuple((p6 + b_vec * scale).astype(int)), (0,255,255), 6)
                cv2.line(vis, tuple(p6.astype(int)), tuple(target.astype(int)), (255,0,0), 6)
                st.session_state['snapshot_result'] = (vis, angle_between(b_vec, curr_vec), 0, "회전 측정"); st.rerun()
            else: st.session_state['error_msg'] = "영상 신호 없음"; st.rerun()
        else: st.session_state['error_msg'] = "먼저 Baseline을 설정하세요."; st.rerun()

    if st.button("▶ 분석 시작", use_container_width=True): pass # WebRTC 자동 실행 중
    if st.button("⏹ 분석 정지", use_container_width=True): pass
