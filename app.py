import os
import json
import base64
import cv2
import numpy as np
import time
import datetime
import joblib
from scipy.spatial.distance import cosine
from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace

app = Flask(__name__, static_folder='.', static_url_path='')

# --- KONFIGURASI PATH ---
TRAINING_DATA_PATH = "training_dataset"
RESEARCH_DATA_PATH = "research_dataset"
ROSTER_PATH = "class_rosters"
STUDENT_INFO_FILE = "student_info.json"
DATABASE_FILE = "face_database.pkl" # Pengganti model SVM
MODEL_CONFIG_FILE = "model_config.json"

os.makedirs(TRAINING_DATA_PATH, exist_ok=True)
os.makedirs(RESEARCH_DATA_PATH, exist_ok=True)
os.makedirs(ROSTER_PATH, exist_ok=True)

# Global Cache
face_database = {} # Format: {'NIM': [vector1, vector2, ...]}
current_model_config = {}
current_active_session = None

# THRESHOLD (Ambang Batas)
# VGG-Face dengan Cosine: < 0.40 adalah orang yang sama
# Semakin kecil angka jarak, semakin mirip.
THRESHOLD_COSINE = 0.40

# --- HELPER FUNCTIONS ---

def b64_to_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(',')[1]
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # FIX: DeepFace butuh RGB, OpenCV baca BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

def load_database():
    global face_database, current_model_config
    try:
        if os.path.exists(DATABASE_FILE):
            face_database = joblib.load(DATABASE_FILE)
            print(f">> Database Wajah Dimuat: {len(face_database)} Mahasiswa")
        
        if os.path.exists(MODEL_CONFIG_FILE):
            with open(MODEL_CONFIG_FILE, 'r') as f:
                current_model_config = json.load(f)
        else:
            current_model_config = {"model_name": "VGG-Face", "detector_backend": "opencv"}
    except Exception as e:
        print(f"Warning: Database belum siap ({e})")

# LOGIKA UTAMA: COSINE SIMILARITY
def find_best_match(target_emb, db):
    best_id = "Unknown"
    min_dist = 100.0 # Mulai dari jarak jauh

    # Linear Search (Loop semua mahasiswa)
    for sid, vectors in db.items():
        for db_vec in vectors:
            dist = cosine(target_emb, db_vec)
            if dist < min_dist:
                min_dist = dist
                best_id = sid
    
    return best_id, min_dist

# --- API ENDPOINTS ---

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/collect_dataset', methods=['POST'])
def api_collect():
    try:
        data = request.json
        sid = data.get('id')
        name = data.get('name')
        images = data.get('images')

        if not all([sid, name, images]): return jsonify(message="Data tidak lengkap", status="error"), 400
        
        folder_name = f"{sid}_{name.replace(' ', '_')}"
        folder_path = os.path.join(TRAINING_DATA_PATH, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        for i, img_b64 in enumerate(images):
            # Simpan sebagai BGR (standar OpenCV file)
            img_rgb = b64_to_image(img_b64)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) 
            cv2.imwrite(os.path.join(folder_path, f"{i+1}.jpg"), img_bgr)
        
        info = load_student_info()
        info[str(sid)] = name
        save_student_info(info)

        return jsonify(message=f"Dataset {name} berhasil disimpan", status="success")
    except Exception as e:
        return jsonify(message=str(e), status="error"), 500

# GANTI DARI TRAINING SVM KE INDEXING DATABASE
@app.route('/api/train_model', methods=['POST'])
def api_train():
    global face_database, current_model_config
    try:
        conf = request.json
        model_name = conf.get('model_name', 'VGG-Face')
        detector = conf.get('detector_backend', 'opencv')
        
        print(f"--- Updating Database: {model_name} + {detector} ---")
        
        new_db = {}
        folders = [f for f in os.listdir(TRAINING_DATA_PATH) if os.path.isdir(os.path.join(TRAINING_DATA_PATH, f))]
        
        if not folders: return jsonify(message="Tidak ada data mahasiswa", status="error"), 400

        for folder in folders:
            # Format Folder: NIM_Nama -> Ambil NIM
            sid = folder.split('_')[0]
            path = os.path.join(TRAINING_DATA_PATH, folder)
            files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
            
            vecs = []
            for f in files:
                try:
                    emb = DeepFace.represent(
                        img_path=os.path.join(path, f),
                        model_name=model_name,
                        detector_backend=detector,
                        enforce_detection=False
                    )[0]["embedding"]
                    vecs.append(emb)
                except: pass
            
            if vecs:
                new_db[sid] = vecs
                print(f"Indexed {sid}: {len(vecs)} vectors")

        face_database = new_db
        joblib.dump(face_database, DATABASE_FILE)
        
        current_model_config = {"model_name": model_name, "detector_backend": detector}
        with open(MODEL_CONFIG_FILE, 'w') as f: json.dump(current_model_config, f)
        
        return jsonify(message=f"Database Wajah Diperbarui ({len(new_db)} Mahasiswa)", status="success")

    except Exception as e:
        return jsonify(message=f"Error Update: {str(e)}", status="error"), 500

# BENCHMARK DENGAN COSINE SIMILARITY (1-Nearest Neighbor)
@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    try:
        conf = request.json
        model_name = conf.get('model_name')
        detector = conf.get('detector_backend')
        source_type = conf.get('dataset_source', 'research')
        
        target_path = TRAINING_DATA_PATH if source_type == 'student' else RESEARCH_DATA_PATH
        if not os.path.exists(target_path):
             return jsonify(message=f"Folder {target_path} tidak ditemukan.", status="error"), 400

        folders = [f for f in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, f))]
        if len(folders) < 2: 
            return jsonify(message="Data kurang (min 2 kelas).", status="error"), 400

        # Kita split: Sebagian jadi Database (Gallery), Sebagian jadi Soal Test (Probe)
        gallery_db = {}
        probes = []
        inference_times = []
        
        print("--- Benchmarking 1-NN Cosine ---")

        for label in folders:
            path = os.path.join(target_path, label)
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png'))][:25]
            
            # Split 80% Database, 20% Test
            split = int(len(files) * 0.8)
            train_files = files[:split]
            test_files = files[split:]

            # Isi Gallery
            vecs = []
            for f in train_files:
                try:
                    e = DeepFace.represent(os.path.join(path, f), model_name=model_name, detector_backend=detector, enforce_detection=False)[0]["embedding"]
                    vecs.append(e)
                except: pass
            if vecs: gallery_db[label] = vecs

            # Isi Test
            for f in test_files:
                try:
                    t0 = time.time()
                    e = DeepFace.represent(os.path.join(path, f), model_name=model_name, detector_backend=detector, enforce_detection=False)[0]["embedding"]
                    t1 = time.time()
                    inference_times.append(t1 - t0)
                    probes.append((label, e))
                except: pass

        if not probes: return jsonify(message="Gagal ekstrak fitur test.", status="error"), 500

        correct = 0
        total_similarity = 0 # Menggantikan confidence

        for true_label, target_vec in probes:
            # Cari match dengan Cosine
            pred_id, dist = find_best_match(target_vec, gallery_db)
            
            # Logic pencocokan label
            # Jika dataset mahasiswa: folder "123_Nama", tapi ID di db "123". Harus disamakan.
            norm_true = true_label.split('_')[0] if source_type == 'student' else true_label
            norm_pred = pred_id.split('_')[0] if source_type == 'student' else pred_id
            
            if norm_true == norm_pred:
                correct += 1
            
            # Similarity score (0-100%)
            sim = max(0, (1 - dist)) 
            total_similarity += sim

        acc = correct / len(probes)
        avg_sim = total_similarity / len(probes)
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return jsonify(
            status="success",
            results={
                "model": model_name,
                "detector": detector,
                "dataset": "Mahasiswa" if source_type == 'student' else "PINS/Riset",
                "accuracy": round(acc * 100, 2),
                "avg_confidence": round(avg_sim * 100, 2), # Dikirim sebagai "confidence" agar tidak error di frontend
                "avg_time": round(avg_time, 4),
                "fps": round(fps, 2),
                "total_samples": len(probes)
            }
        )

    except Exception as e:
        print(e)
        return jsonify(message=f"Benchmark Gagal: {str(e)}", status="error"), 500

# API ABSENSI (COSINE)
@app.route('/api/attend', methods=['POST'])
def api_attend():
    global current_active_session, face_database
    
    if not current_active_session: return jsonify(message="Belum ada sesi aktif.", status="error"), 400
    if not face_database: return jsonify(message="Database kosong. Lakukan Training/Update dulu.", status="error"), 500

    try:
        data = request.json
        model_name = current_model_config.get('model_name', 'VGG-Face')
        
        # Opsi Pilihan User Tetap Ada
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

        # GANTI LOGIKA SVM -> COSINE
        student_id, dist = find_best_match(emb, face_database)
        
        # Ubah Distance ke Similarity % untuk tampilan
        # Distance 0 = 100% mirip. Distance 0.4 = 60% mirip (kasarnya)
        similarity = (1 - dist) * 100
        sim_str = round(similarity, 1)

        # CEK THRESHOLD
        if dist > THRESHOLD_COSINE:
            return jsonify(message=f"Wajah Asing (Mirip: {sim_str}%)", status="error")
        
        info = load_student_info()
        student_name = info.get(student_id, "Unknown")

        att_data = load_attendance_data(current_active_session['course'], current_active_session['class'])
        if student_id not in att_data['students']:
            return jsonify(message=f"{student_name} tidak terdaftar ({sim_str}%)", status="warning")

        active_meet = None
        for m in reversed(att_data['meetings']):
            if m['meeting_number'] == current_active_session['meeting']:
                active_meet = m
                break
        
        if active_meet:
            if not active_meet['attendance'].get(student_id):
                active_meet['attendance'][student_id] = True
                save_attendance_data(current_active_session['course'], current_active_session['class'], att_data)
                return jsonify(message=f"Hadir: {student_name} ({sim_str}%)", status="success")
            else:
                return jsonify(message=f"Sudah Absen: {student_name} ({sim_str}%)", status="info")

        return jsonify(message="Error System", status="error")

    except Exception as e:
        return jsonify(message=f"Error: {str(e)}", status="error"), 500

# Helper Routes (Sama seperti sebelumnya)
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
        # Cek apakah ID ada di database wajah
        if d['id'] not in face_database: return jsonify(message="ID belum registrasi dataset", status="error"), 404
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
    load_database()
    app.run(debug=True, host='0.0.0.0', port=5000)