print("RUNNING THIS APP.PY")
from flask import Flask, request, jsonify, send_file
import os, cv2, sqlite3, json
import numpy as np
from datetime import datetime
import tensorflow as tf
import tf_keras
from tf_keras.models import load_model
import uuid
from rag_service import initialize_rag, get_disease_details
import time
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 
basepat = os.path.dirname(os.path.abspath(__file__))
upload = os.path.join(basepat, "uploads")
model_path = os.path.join(basepat, "unified_model.h5")
class_json_path = os.path.join(basepat, "classes.json")
dbpat = os.path.join(basepat, "database.db")
os.makedirs(upload, exist_ok=True)
unified_model = None
classes_list = []

def load_unified_resources():
    global unified_model, classes_list
    if os.path.exists(model_path) and os.path.exists(class_json_path):
        try:
            unified_model = load_model(model_path)
            with open(class_json_path, 'r') as f:
                class_indices = json.load(f)
                # Sort indices to get correct order
                classes_list = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
            print(f"Unified model loaded with {len(classes_list)} classes.")
        except Exception as e:
            print(f"Error loading unified model: {e}")

load_unified_resources()
initialize_rag()
def init_db():
    conn = sqlite3.connect(dbpat)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop TEXT,
            disease TEXT,
            stage TEXT,
            severity REAL,
            time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()
def calculate_severity(path):
    img = cv2.imread(path)
    if img is None: return 0.0, "Unknown"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    sev = (np.mean(gray) + np.mean(hsv[:,:,1]) + np.mean(lab[:,:,0])) / 3
    sev = min(100, max(0, sev / 2))
    if sev < 30: stage = "Initial"
    elif sev < 60: stage = "Moderate"
    elif sev < 90: stage = "Severe"
    else: stage = "Critical"
    return round(sev,2), stage

def generate_gradcam(model, img):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                last_conv = layer.name
                break   
        if not last_conv: return None
        grad_model = tf_keras.models.Model(
            model.inputs,
            [model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv, preds = grad_model(img)
            idx = tf.argmax(preds[0])
            loss = preds[:, idx]
        grads = tape.gradient(loss, conv)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = conv[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap,0) / tf.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None

def parse_label(label):
    parts = label.split('___')
    raw_crop = parts[0].split('(')[0].split(',')[0].replace('_', ' ').strip().lower()
    crop = raw_crop.split()[0]
    disease = parts[1].replace('_', ' ').strip()
    return crop, disease

@app.route("/predict", methods=["POST"])
def predict():

    try:
        start_time = time.time()
        print("\n DIAGNOSIS START ==========")
        if not unified_model:
            load_unified_resources()
            if not unified_model:
                return jsonify({"error": "Unified model not yet available."}), 503
        
        selected_crop = request.form.get("crop", "").lower().strip()
        file = request.files.get("file")
        if not file: return jsonify({"error": "No file uploaded"}), 400

        filename = f"{uuid.uuid4()}_{file.filename}"
        path = os.path.join(upload, filename)
        file.save(path)
        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "Failed to read uploaded image."}), 400           
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_r = cv2.resize(img_rgb, (224,224))
        img_arr = np.expand_dims(img_r, axis=0) / 255.0
        t = time.time()
        print( f"Model prediction: {time.time() - t:.2f} seconds" )
        preds = unified_model.predict(img_arr)
        confidence = float(np.max(preds))
        label = classes_list[np.argmax(preds)]
        detected_crop, disease = parse_label(label)
        print(f"TRACE: Selected={selected_crop}, Detected={detected_crop}, Confidence={confidence:.4f}")       
        if detected_crop != selected_crop and confidence > 0.4:
            return jsonify({
                "error": f"The selected image is detected as {detected_crop.upper()}, not {selected_crop.upper()}. Please select the correct crop."
            }), 400   
        if confidence < 0.3:
             return jsonify({"error": "Invalid detection. Please use a clear image."}), 400
        if "healthy" in disease.lower():
            severity = 0.0
            stage = "Healthy"
        else:
            t = time.time()
            print(f"Severity calculation: {time.time() - t:.2f} seconds")
            severity, stage = calculate_severity(path)
        t = time.time()
        print( f"Database operation: {time.time() - t:.2f} seconds")
        conn = sqlite3.connect(dbpat)
        cur = conn.cursor()
        cur.execute("INSERT INTO predictions VALUES (NULL,?,?,?,?,?)",
            (detected_crop, disease, stage, severity, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({
            "crop": detected_crop,
            "disease": disease,
            "stage": stage,
            "severity_percent": severity,
            "confidence": confidence,
        })
        print( f"TOTAL DIAGNOSIS TIME: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        import traceback
        print("CRITICAL ERROR IN PREDICT:")
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500
@app.route("/disease-report", methods=["POST"])
def disease_report():
    payload = request.get_json(silent=True) or {}
    crop = str(payload.get("crop", "")).strip()
    disease = str(payload.get("disease", "")).strip()

    if not crop or not disease:
        return jsonify({
            "error": "Both crop and disease are required."
        }), 400

    try:
        question = f"""
        Give complete information about {disease} in {crop}.
        Include:
        Scientific Name
        Symptoms
        Cause
        Favorable Conditions
        Treatment
        Chemical Control
        Organic Control
        Prevention
        """
        search_query = f"{disease} in {crop}"
        answer = get_disease_details(question, search_query)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({
            "error": f"Failed to generate disease report: {str(e)}"
        }), 500

@app.route("/disease-details", methods=["POST"])
def disease_details():
    question = request.json["question"]
    answer = get_disease_details(question)
    return jsonify({"answer": answer})
@app.route("/gradcam", methods=["POST"])
def create_gradcam():
    file = request.files["file"]
    path = os.path.join(upload, file.filename)
    file.save(path)
    img = cv2.imread(path)
    if img is None:
        return jsonify({
            "error": "Could not read image"
        }), 400
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(
        img_rgb,
        (224, 224)
    )
    img_arr = np.expand_dims(
        img_r,
        axis=0
    ) / 255.0
    heatmap = generate_gradcam(
        unified_model,
        img_arr
    )
    output_path = os.path.join(
        upload,
        "gradcam_" + file.filename
    )
    cv2.imwrite(
        output_path,
        heatmap
    )
    return jsonify({
        "gradcam": output_path
    })
@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "model_loaded": unified_model is not None,
        "classes_count": len(classes_list)
    })

@app.route("/")
def home():
    return "Plant Disease Detector Unified Backend is running. Access /health for status."
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)

