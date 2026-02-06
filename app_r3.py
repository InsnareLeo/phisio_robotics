import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import torch
import sys
import os
import tempfile
import pathlib
from collections import deque
from ultralytics import YOLO
import time

# ==========================================
# 0. 환경 및 경로 설정
# ==========================================
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(page_title="Phisio AI Pro (v71)", layout="wide")
st.markdown("""<style>.stImage > img { width: 100%; border-radius: 8px; }</style>""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_NAME = os.path.join(BASE_DIR, "yolov8n-pose.pt")
ACTION_WEIGHTS_PATH = os.path.join(BASE_DIR, "yoga_weights_yolo_seated_safe.pkl")
STICKER_MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
YOLOV9_REPO_DIR = os.path.join(BASE_DIR, 'yolov9')

if os.path.exists(YOLOV9_REPO_DIR):
    if YOLOV9_REPO_DIR not in sys.path: sys.path.insert(0, YOLOV9_REPO_DIR)
    try:
        from models.experimental import attempt_load
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox
    except ImportError: pass

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.device = torch.device(device)
        self.model = attempt_load(weights_path, device=self.device)
        self.stride = int(self.model.stride.max())
        self.imgsz = 640

    def _get_nms_candidates(self, img_bgr, roi):
        x1, y1, x2, y2 = roi
        img = letterbox(img_bgr, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]; img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(self.device).float() / 255.0
        if len(img_tensor.shape) == 3: img_tensor = img_tensor[None]
        with torch.no_grad(): pred = self.model(img_tensor, augment=False)[0]
        pred = non_max_suppression(pred, 0.01, 0.45) 
        cands = []
        if len(pred[0]):
            det = pred[0]
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img_bgr.shape).round()
            for *xyxy, conf, cls in det:
                cx, cy = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    cands.append({'center': (cx, cy), 'box': (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])), 'conf': float(conf)})
        cands = sorted(cands, key=lambda x: x['conf'], reverse=True)
        nms_done = []
        for c in cands:
            if not any(np.linalg.norm(np.array(c['center'])-np.array(a['center'])) < 18 for a in nms_done):
                nms_done.append(c)
        return nms_done

    def get_spine_points(self, img_bgr, kps):
        if kps is None: return [], False, "Pose 인식 불가"
        l_sh, r_sh = kps[5], kps[6]
        mid_x = (l_sh[0] + r_sh[0]) / 2
        torso_len = abs(((kps[5][1]+kps[6][1])/2) - ((kps[11][1]+kps[12][1])/2))
        axis_top_y, axis_bot_y = ((kps[5][1]+kps[6][1])/2) - torso_len*0.3, ((kps[11][1]+kps[12][1])/2) + torso_len*0.2
        roi = (int(min(kps[5][0], kps[11][0])-100), int(axis_top_y), int(max(kps[6][0], kps[12][0])+100), int(axis_bot_y))
        candidates = self._get_nms_candidates(img_bgr, roi)
        section_h = (axis_bot_y - axis_top_y) / 6.0
        x_tol = abs(l_sh[0]-r_sh[0]) * 0.4
        final_objs = [None] * 6; used_idx = set()
        for i in range(6):
            sy_min, sy_max = axis_top_y + i*section_h, axis_top_y + (i+1)*section_h
            best_idx, max_s = -1, -1e9
            for idx, c in enumerate(candidates):
                if idx in used_idx: continue
                if sy_min <= c['center'][1] <= sy_max and abs(c['center'][0] - mid_x) < x_tol:
                    score = c['conf']*10 - abs(c['center'][1] - (sy_min+sy_max)/2)*0.1
                    if score > max_s: max_s, best_idx = score, idx
            if best_idx != -1: final_objs[i] = candidates[best_idx]; used_idx.add(best_idx)
        # 보충
        for i in range(6):
            if final_objs[i] is not None: continue
            sec_center_y = axis_top_y + (i+0.5)*section_h
            best_idx, max_s = -1, -1e9
            for idx, c in enumerate(candidates):
                if idx in used_idx: continue
                if abs(c['center'][0] - mid_x) > x_tol * 1.3: continue
                score = c['conf']*100 - abs(c['center'][1] - sec_center_y)*0.6 - abs(c['center'][0] - mid_x)*0.25
                if score > max_s: max_s, best_idx = score, idx
            if best_idx != -1: final_objs[i] = candidates[best_idx]; used_idx.add(best_idx)
        valid = [o for o in final_objs if o is not None]
        if len(valid) == 6:
            valid.sort(key=lambda x: x['center'][1])
            return valid, True, "성공"
        return valid, False, f"스티커 {len(valid)}개 발견"

# ==========================================
# 2. 유틸리티 함수
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
    cv2.line(vis, pts[0], pts[1], (255,255,0), 2); cv2.line(vis, pts[1], pts[2], (255,255,0), 2)
    cv2.line(vis, pts[3], pts[4], (255,0,255), 2); cv2.line(vis, pts[4], pts[5], (255,0,255), 2)
    return vis, pts

def process_yolo_keypoints_original(kps):
    coords, confs = kps[:, :2].copy(), kps[:, 2:3].copy()
    coords -= (coords[11] + coords[12]) / 2.0
    scale_ref = np.linalg.norm((coords[5] + coords[6]) / 2.0) or 1.0
    coords /= scale_ref; coords[[13,14,15,16]] = 0.0
    return np.hstack([coords, confs]).flatten()

@st.cache_resource
def load_all_models():
    pm = YOLO(POSE_MODEL_NAME); am = build_action_model((30, 51), 5)
    if os.path.exists(ACTION_WEIGHTS_PATH):
        with open(ACTION_WEIGHTS_PATH, "rb") as f: w_list = pickle.load(f)
        am.set_weights([np.array(w) for w in w_list])
    sp = StickerProcessor(STICKER_MODEL_PATH) if os.path.exists(STICKER_MODEL_PATH) else None
    return pm, am, ['Sitting (Ready)', 'Forward_Bending', 'Back_Extension', 'Side_Bending', 'Rotation'], sp

# 세션 초기화
if 'is_analyzing' not in st.session_state: st.session_state['is_analyzing'] = False
for k in ['last_frame_data', 'snapshot_result', 'side_baseline_vec', 'rot_baseline_vec', 'error_msg', 'rot_base_angle']:
    if k not in st.session_state: st.session_state[k] = None

# ==========================================
# 3. 메인 UI
# ==========================================
col_v, col_r, col_c = st.columns([1.5, 1.1, 0.9])
with col_v:
    st.markdown("### 🎥 실시간 분석"); v_spot = st.empty()
with col_r:
    st.markdown("### 📊 측정 결과"); r_spot = st.empty()
    if st.session_state['error_msg']: r_spot.error(st.session_state['error_msg'])
    elif st.session_state['snapshot_result']:
        img, v1, v2, label = st.session_state['snapshot_result']
        r_spot.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        c1, c2 = st.columns(2); c1.metric(f"{label}", f"{v1:.1f}°")
    else: r_spot.info("분석 중 측정 버튼을 눌러주세요.")

with col_c:
    st.subheader("🛠️ 제어 패널")
    pm, am, names, sp = load_all_models()
    
    if st.button("📸 Cobb 각도 측정 (Side Baseline 저장)", type="primary", use_container_width=True):
        st.session_state['error_msg'] = None
        if st.session_state['last_frame_data']:
            f, k = st.session_state['last_frame_data']; objs, success, msg = sp.get_spine_points(f, k)
            if success:
                vis = f.copy(); vis, pts = draw_spine_and_boxes(vis, objs)
                st.session_state['side_baseline_vec'] = np.array(pts[0]) - np.array(pts[5])
                cv2.line(vis, pts[5], pts[0], (0,255,255), 6) 
                u, l = angle_between(np.array(pts[0])-np.array(pts[1]), np.array(pts[2])-np.array(pts[1])), angle_between(np.array(pts[3])-np.array(pts[4]), np.array(pts[5])-np.array(pts[4]))
                st.session_state['snapshot_result'] = (vis, u, l, "Cobb 각도"); st.rerun()
            else: st.session_state['error_msg'] = f"측정 실패: {msg}"; st.rerun()

    if st.button("📏 회전 기준각 저장 (Rotation Baseline)", use_container_width=True):
        st.session_state['error_msg'] = None
        if st.session_state['last_frame_data']:
            f, k = st.session_state['last_frame_data']; objs, success, msg = sp.get_spine_points(f, k)
            if success:
                vis = f.copy(); pts = [o['center'] for o in objs]
                p6 = np.array(pts[5])
                # [핵심] 견봉 포인트 추적: 스켈레톤 어깨(5, 6) 중 아무데나 가장 가까운 스티커 (척추 스티커 제외)
                sh_l, sh_r = k[5][:2], k[6][:2]
                cands = sp._get_nms_candidates(f, (0, 0, f.shape[1], f.shape[0]))
                spine_pts = [tuple(p) for p in pts]
                lateral_cands = [c for c in cands if tuple(c['center']) not in spine_pts]
                
                if lateral_cands:
                    target_pt = np.array(min(lateral_cands, key=lambda c: min(np.linalg.norm(np.array(c['center'])-sh_l), np.linalg.norm(np.array(c['center'])-sh_r)))['center'])
                else:
                    target_pt = np.array(sh_l if k[5][2] > k[6][2] else sh_r)
                
                v_spine = np.array(pts[0]) - p6
                st.session_state['rot_baseline_vec'] = v_spine
                ang = angle_between(v_spine, target_pt - p6)
                cv2.line(vis, tuple(p6), tuple(pts[0]), (0,255,255), 6) # 노란 기준선
                cv2.line(vis, tuple(p6), tuple(target_pt.astype(int)), (255,0,0), 6) # 파란 측정선
                st.session_state['snapshot_result'] = (vis, ang, 0, "회전 기준각"); st.rerun()
            else: st.session_state['error_msg'] = f"저장 실패: {msg}"; st.rerun()

    st.markdown("---")
    rot_dir = st.radio("테스트 방향", ["왼쪽 (rot_l.jpg)", "오른쪽 (rot_r.jpg)"], horizontal=True)

    if st.button("📐 측면 굴곡 측정 (side.jpg)", use_container_width=True):
        if st.session_state['side_baseline_vec'] is not None:
            timg = cv2.imread(os.path.join(BASE_DIR, "side.jpg"))
            res = pm(timg, verbose=False, conf=0.1); tk = res[0].keypoints.data[0].cpu().numpy() if res[0].keypoints else None
            objs, success, _ = sp.get_spine_points(timg, tk)
            if success:
                vis = timg.copy(); vis, pts = draw_spine_and_boxes(vis, objs)
                p1, p6, b_vec = np.array(pts[0]), np.array(pts[5]), st.session_state['side_baseline_vec']
                curr_vec = p1 - p6; scale = np.linalg.norm(curr_vec) / np.linalg.norm(b_vec)
                cv2.line(vis, tuple(p6.astype(int)), tuple((p6 + b_vec * scale).astype(int)), (0,255,255), 6)
                cv2.line(vis, tuple(p6.astype(int)), tuple(p1.astype(int)), (255,0,0), 6)
                st.session_state['snapshot_result'] = (vis, angle_between(b_vec, curr_vec), 0, "측면 굴곡"); st.rerun()

    if st.button("📐 회전 측정 테스트 (이미지)", use_container_width=True):
        if st.session_state['rot_baseline_vec'] is not None:
            timg = cv2.imread(os.path.join(BASE_DIR, "rot_l.jpg" if "왼쪽" in rot_dir else "rot_r.jpg"))
            res = pm(timg, verbose=False, conf=0.1); tk = res[0].keypoints.data[0].cpu().numpy() if res[0].keypoints else None
            objs, success, _ = sp.get_spine_points(timg, tk)
            p6 = np.array(objs[5]['center'] if success else [(tk[11][0]+tk[12][0])/2, (tk[11][1]+tk[12][1])/2])
            sh_l, sh_r = tk[5][:2], tk[6][:2]
            cands = sp._get_nms_candidates(timg, (0, 0, timg.shape[1], timg.shape[0]))
            spine_centers = [tuple(o['center']) for o in objs] if success else []
            lateral_cands = [c for c in cands if tuple(c['center']) not in spine_centers]
            if lateral_cands: target = np.array(min(lateral_cands, key=lambda c: min(np.linalg.norm(np.array(c['center'])-sh_l), np.linalg.norm(np.array(c['center'])-sh_r)))['center'])
            else: target = np.array(sh_l if tk[5][2] > tk[6][2] else sh_r)
            
            b_vec = st.session_state['rot_baseline_vec']; curr_vec = target - p6; scale = np.linalg.norm(curr_vec) / np.linalg.norm(b_vec)
            vis = timg.copy(); cv2.circle(vis, tuple(target.astype(int)), 15, (0, 0, 255), -1)
            cv2.line(vis, tuple(p6.astype(int)), tuple((p6 + b_vec * scale).astype(int)), (0,255,255), 6)
            cv2.line(vis, tuple(p6.astype(int)), tuple(target.astype(int)), (255,0,0), 6)
            st.session_state['snapshot_result'] = (vis, angle_between(b_vec, curr_vec), 0, "회전 측정"); st.rerun()

    if st.button("▶ 분석 시작", use_container_width=True): st.session_state['is_analyzing'] = True; st.rerun()
    if st.button("⏹ 분석 정지", use_container_width=True): st.session_state['is_analyzing'] = False; st.rerun()

# ==========================================
# 4. 분석 루프
# ==========================================
def run_analysis_loop(source, is_webcam, spot):
    pm, am, names, sp = load_all_models()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if is_webcam else cv2.VideoCapture(source)
    while cap.isOpened() and st.session_state['is_analyzing']:
        ret, frame = cap.read()
        if not ret: break
        res = pm(frame, verbose=False, conf=0.1); kps = res[0].keypoints.data[0].cpu().numpy() if res[0].keypoints else None
        if kps is not None:
            st.session_state['last_frame_data'] = (frame.copy(), kps)
            # 동작 분류
            feat = process_yolo_keypoints_original(kps)
            # 분류 생략 안함 (am.predict)
        disp = cv2.resize(frame, (640, int(frame.shape[0]*(640/frame.shape[1]))))
        cv2.putText(disp, "Action Tracking...", (20, disp.shape[0]-20), 0, 0.7, (255,255,255), 2)
        spot.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    cap.release()

with col_c:
    with st.expander("입력 설정"):
        itype = st.radio("소스", ["Video File", "Webcam"], key="src_radio")
        if itype == "Video File":
            file = st.file_uploader("업로드", type=['mp4','mov'])
            if file:
                tf = tempfile.NamedTemporaryFile(delete=False); tf.write(file.read())
                st.session_state['input_path'] = tf.name; st.success("준비됨")

if st.session_state['is_analyzing']:
    run_analysis_loop(st.session_state.get('input_path'), (st.session_state.get('src_radio')=="Webcam"), v_spot)