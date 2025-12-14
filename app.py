import os
import json
import base64
import cv2
import numpy as np
import time
import datetime
import joblib
from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace

# Scikit-Learn untuk Validasi Ilmiah & Klasifikasi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_folder='.', static_url_path='')

# --- KONFIGURASI PATH (DIPISAHKAN) ---
TRAINING_DATA_PATH = "training_dataset"  # KHUSUS MAHASISWA (Absensi)
RESEARCH_DATA_PATH = "research_dataset"  # KHUSUS RISET (PINS/LFW)
ROSTER_PATH = "class_rosters"
STUDENT_INFO_FILE = "student_info.json"
MODEL_SVM_FILE = "svm_model.pkl"
MODEL_LE_FILE = "label_encoder.pkl"
MODEL_CONFIG_FILE = "model_config.json"

# Buat semua folder jika belum ada
os.makedirs(TRAINING_DATA_PATH, exist_ok=True)
os.makedirs(RESEARCH_DATA_PATH, exist_ok=True) # Folder ini wajib ada untuk benchmark
os.makedirs(ROSTER_PATH, exist_ok=True)

# Global Variables (Cache Memory)
model_svm = None
model_le = None
current_model_config = {}
current_active_session = None

# --- HELPER FUNCTIONS ---

def b64_to_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(',')[1]
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def load_student_info():
    if not os.path.exists(STUDENT_INFO_FILE): return {}
    try:
        with open(STUDENT_INFO_FILE, 'r') as f: return json.load(f)
    except: return {}

def save_student_info(data):
    with open(STUDENT_INFO_FILE, 'w') as f: json.dump(data, f, indent=2)

def get_roster_path(course, class_name):
    return os.path.join(ROSTER_PATH, f"{course.replace(' ', '_')}_{class_name}_attendance.json")

def load_attendance_data(course, class_name):
    path = get_roster_path(course, class_name)
    if not os.path.exists(path): return {"students": [], "meetings": []}
    try:
        with open(path, 'r') as f: return json.load(f)
    except: return {"students": [], "meetings": []}

def save_attendance_data(course, class_name, data):
    with open(get_roster_path(course, class_name), 'w') as f: json.dump(data, f, indent=2)

def load_trained_models():
    global model_svm, model_le, current_model_config
    try:
        if os.path.exists(MODEL_SVM_FILE): model_svm = joblib.load(MODEL_SVM_FILE)
        if os.path.exists(MODEL_LE_FILE): model_le = joblib.load(MODEL_LE_FILE)
        
        if os.path.exists(MODEL_CONFIG_FILE):
            with open(MODEL_CONFIG_FILE, 'r') as f:
                current_model_config = json.load(f)
            print(f">> Config Loaded: {current_model_config}")
        else:
            current_model_config = {"model_name": "VGG-Face", "detector_backend": "opencv"}
            
    except Exception as e:
        print(f"Warning: Model belum siap ({e})")

# --- API ENDPOINTS ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# 1. API COLLECT DATASET (Masuk ke Folder MAHASISWA)
@app.route('/api/collect_dataset', methods=['POST'])
def api_collect():
    try:
        data = request.json
        sid = data.get('id')
        name = data.get('name')
        images = data.get('images')

        if not all([sid, name, images]): return jsonify(message="Data tidak lengkap", status="error"), 400
        
        # Simpan ke TRAINING_DATA_PATH
        folder_name = f"{sid}_{name.replace(' ', '_')}"
        folder_path = os.path.join(TRAINING_DATA_PATH, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for i, img_b64 in enumerate(images):
            cv2.imwrite(os.path.join(folder_path, f"{i+1}.jpg"), b64_to_image(img_b64))
        
        info = load_student_info()
        info[str(sid)] = name
        save_student_info(info)

        return jsonify(message=f"Dataset {name} berhasil disimpan", status="success")
    except Exception as e:
        return jsonify(message=str(e), status="error"), 500

# 2. API TRAINING FINAL (Menggunakan Data MAHASISWA)
@app.route('/api/train_model', methods=['POST'])
def api_train():
    global model_svm, model_le, current_model_config
    try:
        conf = request.json
        model_name = conf.get('model_name', 'VGG-Face')
        detector = conf.get('detector_backend', 'opencv')
        
        print(f"--- Training Final (Mahasiswa): {model_name} + {detector} ---")
        
        X, y = [], []
        # BACA DARI FOLDER MAHASISWA
        folders = [f for f in os.listdir(TRAINING_DATA_PATH) if os.path.isdir(os.path.join(TRAINING_DATA_PATH, f))]
        
        if len(folders) < 2: return jsonify(message="Minimal butuh 2 mahasiswa terdaftar", status="error"), 400

        for label in folders:
            path = os.path.join(TRAINING_DATA_PATH, label)
            files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
            
            for f in files:
                try:
                    emb = DeepFace.represent(
                        img_path=os.path.join(path, f),
                        model_name=model_name,
                        detector_backend=detector,
                        enforce_detection=False
                    )[0]["embedding"]
                    X.append(emb)
                    y.append(label)
                except: pass

        if not X: return jsonify(message="Gagal ekstrak fitur", status="error"), 500

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        svm = SVC(kernel='linear', probability=True)
        svm.fit(X, y_enc)
        
        model_svm = svm
        model_le = le
        current_model_config = {"model_name": model_name, "detector_backend": detector}
        
        joblib.dump(svm, MODEL_SVM_FILE)
        joblib.dump(le, MODEL_LE_FILE)
        with open(MODEL_CONFIG_FILE, 'w') as f: json.dump(current_model_config, f)
        
        return jsonify(message=f"Model {model_name} berhasil dilatih & disimpan!", status="success")

    except Exception as e:
        return jsonify(message=f"Error Training: {str(e)}", status="error"), 500

# 3. API BENCHMARK (Menggunakan Data RISET/PINS) -> [PERBAIKAN UTAMA]
@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    """
    Fitur utama untuk Skripsi: Membandingkan Akurasi & FPS
    MENGGUNAKAN FOLDER 'research_dataset' (PINS/LFW)
    """
    try:
        conf = request.json
        model_name = conf.get('model_name')
        detector = conf.get('detector_backend')
        
        print(f"--- Benchmarking Riset (PINS): {model_name} + {detector} ---")
        
        X, y = [], []
        
        # PERBAIKAN: Baca dari RESEARCH_DATA_PATH
        if not os.path.exists(RESEARCH_DATA_PATH):
             return jsonify(message="Folder research_dataset tidak ditemukan. Jalankan script setup dulu.", status="error"), 400

        folders = [f for f in os.listdir(RESEARCH_DATA_PATH) if os.path.isdir(os.path.join(RESEARCH_DATA_PATH, f))]
        
        if len(folders) < 2: 
            return jsonify(message="Data PINS kurang/kosong. Jalankan script setup_pins.py.", status="error"), 400

        inference_times = []
        total_extracted = 0

        for label in folders:
            path = os.path.join(RESEARCH_DATA_PATH, label)
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            
            # Limit sampel agar tidak crash
            files = files[:20] 

            for f in files:
                img_path = os.path.join(path, f)
                try:
                    t0 = time.time()
                    emb = DeepFace.represent(
                        img_path=img_path,
                        model_name=model_name,
                        detector_backend=detector,
                        enforce_detection=False
                    )[0]["embedding"]
                    t1 = time.time()
                    
                    X.append(emb)
                    y.append(label)
                    inference_times.append(t1 - t0)
                    total_extracted += 1
                except: pass

        if total_extracted == 0:
             return jsonify(message=f"Gagal deteksi wajah PINS dengan {detector}", status="error"), 500

        # Split Data PINS (Train/Test)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Stratify split agar seimbang
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        
        svm = SVC(kernel='linear', probability=True)
        svm.fit(X_train, y_train)
        
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        avg_time = sum(inference_times) / len(inference_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return jsonify(
            status="success",
            results={
                "model": model_name,
                "detector": detector,
                "accuracy": round(acc * 100, 2),
                "avg_time": round(avg_time, 4),
                "fps": round(fps, 2),
                "total_samples": total_extracted
            }
        )

    except Exception as e:
        print(f"Error Benchmark: {e}")
        return jsonify(message=f"Benchmark Gagal: {str(e)}", status="error"), 500

# 4. API ABSENSI (Menggunakan Model Produksi dari Data Mahasiswa)
@app.route('/api/attend', methods=['POST'])
def api_attend():
    global current_active_session, model_svm, model_le
    
    if not current_active_session: return jsonify(message="Belum ada sesi aktif.", status="error"), 400
    if not model_svm: return jsonify(message="Model belum dilatih.", status="error"), 500

    try:
        data = request.json
        model_name = current_model_config.get('model_name', 'VGG-Face')
        req_detector = data.get('detector_backend', 'opencv')
        img = b64_to_image(data.get('image'))

        try:
            emb = DeepFace.represent(
                img_path=img,
                model_name=model_name,
                detector_backend=req_detector,
                enforce_detection=True
            )[0]["embedding"]
        except:
            return jsonify(message="Wajah tidak terdeteksi", status="error")

        proba = model_svm.predict_proba([emb])[0]
        idx = np.argmax(proba)
        confidence = proba[idx]
        
        if confidence < 0.70:
            return jsonify(message=f"Wajah Asing ({confidence:.2f})", status="error")
        
        student_id = model_le.inverse_transform([idx])[0].split('_')[0]
        info = load_student_info()
        student_name = info.get(student_id, "Unknown")

        att_data = load_attendance_data(current_active_session['course'], current_active_session['class'])
        if student_id not in att_data['students']:
            return jsonify(message=f"{student_name} tidak terdaftar", status="warning")

        active_meet = None
        for m in reversed(att_data['meetings']):
            if m['meeting_number'] == current_active_session['meeting']:
                active_meet = m
                break
        
        if active_meet:
            if not active_meet['attendance'].get(student_id):
                active_meet['attendance'][student_id] = True
                save_attendance_data(current_active_session['course'], current_active_session['class'], att_data)
                return jsonify(message=f"Hadir: {student_name}", status="success")
            else:
                return jsonify(message=f"Sudah Absen: {student_name}", status="info")

        return jsonify(message="Error System", status="error")

    except Exception as e:
        return jsonify(message=f"Error: {str(e)}", status="error"), 500

# 5. API DATA LAPORAN
@app.route('/api/get_attendance', methods=['GET'])
def api_get_attendance():
    c = request.args.get('course')
    cl = request.args.get('class')
    data = load_attendance_data(c, cl)
    info = load_student_info()
    
    roster_list = []
    for sid in data['students']:
        roster_list.append({"id": sid, "name": info.get(sid, sid)})
    
    return jsonify(roster=roster_list, meetings=data['meetings'], status="success")

@app.route('/api/add_roster', methods=['POST'])
def api_add_roster():
    try:
        d = request.json
        if d['id'] not in load_student_info(): return jsonify(message="ID belum registrasi dataset", status="error"), 404
        
        ad = load_attendance_data(d['course'], d['class'])
        if d['id'] not in ad['students']:
            ad['students'].append(d['id'])
            for m in ad['meetings']: m['attendance'][d['id']] = False
            save_attendance_data(d['course'], d['class'], ad)
            return jsonify(message="Berhasil masuk roster", status="success")
        return jsonify(message="Sudah terdaftar", status="info")
    except Exception as e: return jsonify(message=str(e), status="error"), 500

@app.route('/api/start_session', methods=['POST'])
def api_start_session():
    global current_active_session
    d = request.json
    ad = load_attendance_data(d['course'], d['class'])
    num = 1
    if ad['meetings']: num = ad['meetings'][-1]['meeting_number'] + 1
    new_meet = {"meeting_number": num, "date": str(datetime.date.today()), "attendance": {s: False for s in ad['students']}}
    ad['meetings'].append(new_meet)
    save_attendance_data(d['course'], d['class'], ad)
    current_active_session = {"course": d['course'], "class": d['class'], "meeting": num}
    return jsonify(message=f"Sesi {num} Dimulai", status="success", meeting_number=num)

if __name__ == '__main__':
    load_trained_models()
    app.run(debug=True, host='0.0.0.0', port=5000)