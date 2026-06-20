import os
import sqlite3
import traceback
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from project_config import (
    ANEMIA_CLASS_NAMES,
    CATARACT_CLASS_NAMES,
    ANEMIA_LEGACY_ARTIFACTS,
    ANEMIA_MODEL_CANDIDATES,
    CATARACT_LEGACY_ARTIFACTS,
    CATARACT_MODEL_CANDIDATES,
    DATABASE_PATH,
    MODELS_DIR,
    UPLOADS_DIR,
    UPLOADS_WEB_DIR,
    ensure_runtime_directories,
    list_existing_paths,
    resolve_first_existing,
)

app = Flask(__name__)
app.secret_key = os.getenv("MEDISCAN_SECRET_KEY", "dev-secret-key-change-me")

ensure_runtime_directories()
app.config["UPLOAD_FOLDER"] = str(UPLOADS_DIR)
app.config["UPLOAD_FOLDER_WEB"] = UPLOADS_WEB_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}

anemia_model = None
cataract_model = None
loaded_model_paths = {"anemia": None, "cataract": None}


def init_db():
    """Initialize the local user database if it does not already exist."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS users
               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as exc:
        print(f"Database initialization error: {exc}")


def get_db_connection():
    init_db()
    return sqlite3.connect(DATABASE_PATH)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file_storage):
    """Persist uploads with a unique name and return filesystem/web paths."""
    filename = secure_filename(file_storage.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{filename}"
    filesystem_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    web_path = f"{app.config['UPLOAD_FOLDER_WEB']}/{unique_name}".replace("\\", "/")
    file_storage.save(filesystem_path)
    return filesystem_path, web_path


def load_tf_model(model_label, candidate_paths, preprocess_fn=None, compile_model=False):
    """Load a TensorFlow model from the first canonical artifact that exists."""
    print(f"\nLoading {model_label} model")
    print(f"Canonical candidates: {[str(path) for path in candidate_paths]}")

    model_path = resolve_first_existing(candidate_paths)
    if model_path is None:
        print(f"{model_label} model not found in canonical locations.")
        return None, None

    file_size = model_path.stat().st_size / (1024 * 1024)
    print(f"Found {model_label} model at {model_path} ({file_size:.2f} MB)")

    model = tf.keras.models.load_model(model_path, compile=compile_model)
    print(f"{model_label} model loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
    if preprocess_fn is not None:
        test_input = preprocess_fn(test_input)
    test_output = model.predict(test_input, verbose=0)
    print(f"Smoke-test output shape: {test_output.shape}")

    return model, str(model_path)


def load_models(force_reload=False):
    """Load both models from canonical runtime locations."""
    global anemia_model, cataract_model, loaded_model_paths

    print("\n" + "=" * 60)
    print("Loading models")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}")
    print(f"Models directory exists: {MODELS_DIR.exists()}")
    if MODELS_DIR.exists():
        print(f"Models directory contents: {sorted(os.listdir(MODELS_DIR))}")
    print(f"Legacy anemia artifacts found: {list_existing_paths(ANEMIA_LEGACY_ARTIFACTS)}")
    print(f"Legacy cataract artifacts found: {list_existing_paths(CATARACT_LEGACY_ARTIFACTS)}")

    if force_reload or anemia_model is None:
        try:
            anemia_model, loaded_model_paths["anemia"] = load_tf_model(
                "Anemia",
                ANEMIA_MODEL_CANDIDATES,
                preprocess_fn=tf.keras.applications.mobilenet_v3.preprocess_input,
                compile_model=False,
            )
        except Exception as exc:
            print(f"Error loading anemia model: {exc}")
            print(traceback.format_exc())
            anemia_model = None
            loaded_model_paths["anemia"] = None

    if force_reload or cataract_model is None:
        try:
            cataract_model, loaded_model_paths["cataract"] = load_tf_model(
                "Cataract",
                CATARACT_MODEL_CANDIDATES,
                compile_model=False,
            )
        except Exception as exc:
            print(f"Error loading cataract model: {exc}")
            print(traceback.format_exc())
            cataract_model = None
            loaded_model_paths["cataract"] = None

    print("=" * 60 + "\n")
    return anemia_model is not None, cataract_model is not None


def ensure_models_loaded():
    """Lazy-load models so the app also works when imported by a WSGI server."""
    if anemia_model is None or cataract_model is None:
        return load_models()
    return True, True


def preprocess_image_for_anemia(image_path, target_size=(224, 224)):
    """Preprocess image for the anemia model."""
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as exc:
        print(f"Error preprocessing anemia image: {exc}")
        print(traceback.format_exc())
        return None


from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def preprocess_image_for_cataract(image_path, target_size=(224, 224)):
    """Preprocess image for the cataract model."""
    try:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize(target_size)
        img_array = np.array(img).astype(np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = efficientnet_preprocess(img_array)  # ← THIS LINE WAS MISSING
        return img_array
    except Exception as exc:
        print(f"Error preprocessing cataract image: {exc}")
        print(traceback.format_exc())
        return None

def predict_anemia(image_path):
    """Run anemia inference against the canonical TensorFlow model."""
    global anemia_model
    ensure_models_loaded()

    if anemia_model is None:
        return "Model not loaded", 0.0, "Anemia model is not available. Check server logs."

    try:
        img_array = preprocess_image_for_anemia(image_path)
        if img_array is None:
            return "Error", 0.0, "Failed to process image"

        predictions = anemia_model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_class = ANEMIA_CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions[0]) * 100)
        return predicted_class, confidence, None
    except Exception as exc:
        print(f"Error during anemia prediction: {exc}")
        print(traceback.format_exc())
        return "Error", 0.0, str(exc)


def predict_cataract(image_path):
    """Run cataract inference against the canonical TensorFlow model."""
    global cataract_model
    ensure_models_loaded()

    if cataract_model is None:
        return "Model not loaded", 0.0, "Cataract model is not available. Check server logs."

    try:
        img_array = preprocess_image_for_cataract(image_path)
        if img_array is None:
            return "Error", 0.0, "Failed to process image"

        predictions = cataract_model.predict(img_array, verbose=0)
        print(f"[DEBUG] Raw scores -> normal: {predictions[0][0]:.4f}, cataract: {predictions[0][1]:.4f}")
        CATARACT_THRESHOLD = 0.40
        cataract_score = float(predictions[0][1])
        class_idx = 1 if cataract_score > CATARACT_THRESHOLD else 0
        print(f"[DEBUG] class_idx: {class_idx} -> {'Normal' if class_idx == 0 else 'Cataract'} (threshold={CATARACT_THRESHOLD})")
        confidence = float(cataract_score * 100) if class_idx == 1 else float(predictions[0][0] * 100)
        result = "Normal" if class_idx == 0 else "Cataract"
        return result, confidence, None
    except Exception as exc:
        print(f"Error during cataract prediction: {exc}")
        print(traceback.format_exc())
        return "Error", 0.0, str(exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form.get("email", "")

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, hashed_password, email),
            )
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
            return redirect(url_for("register"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password!", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])


@app.route("/anemia", methods=["GET", "POST"])
def anemia():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected!", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filepath, image_path = save_uploaded_file(file)
            result, confidence, error = predict_anemia(filepath)

            if error:
                flash(f"Error: {error}", "danger")
                return redirect(request.url)

            recommendations = get_anemia_recommendations(result)
            return render_template(
                "result.html",
                disease_type="Anemia",
                result=result,
                confidence=confidence,
                image_path=image_path,
                recommendations=recommendations,
            )

        flash("Invalid file type! Please upload an image.", "danger")
        return redirect(request.url)

    return render_template("anemia.html")


@app.route("/cataract", methods=["GET", "POST"])
def cataract():
    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected!", "danger")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filepath, image_path = save_uploaded_file(file)
            result, confidence, error = predict_cataract(filepath)

            if error:
                flash(f"Error: {error}", "danger")
                return redirect(request.url)

            recommendations = get_cataract_recommendations(result)
            return render_template(
                "result.html",
                disease_type="Cataract",
                result=result,
                confidence=confidence,
                image_path=image_path,
                recommendations=recommendations,
            )

        flash("Invalid file type! Please upload an image.", "danger")
        return redirect(request.url)

    return render_template("cataract.html")


def get_anemia_recommendations(result):
    """Return guidance for the anemia result page."""
    if result == "Anemia":
        return {
            "severity": "Moderate to High",
            "detection_stage": "Early Detection",
            "precautions": [
                "Consult a doctor for confirmatory blood tests.",
                "Take iron supplements only if prescribed by a doctor.",
                "Increase iron-rich foods such as spinach, red meat, beans, and lentils.",
                "Include vitamin C rich foods to support iron absorption.",
                "Avoid tea or coffee with meals because they can reduce iron absorption.",
            ],
            "preventive_measures": [
                "Eat iron-fortified cereals and grains.",
                "Include lean meat, fish, and leafy greens in your diet.",
                "Schedule regular blood tests every three to six months.",
                "Maintain a balanced diet with adequate vitamins.",
                "Exercise regularly to support circulation and overall health.",
            ],
            "treatment_timeline": [
                {"phase": "Doctor consultation and blood test", "duration": "Week 1"},
                {"phase": "Iron supplementation and diet changes", "duration": "Week 2-6"},
                {"phase": "Follow-up assessment", "duration": "Week 6-8"},
                {"phase": "Regular monitoring", "duration": "Ongoing"},
            ],
        }

    return {
        "severity": "Normal",
        "detection_stage": "Healthy",
        "precautions": [
            "Maintain a balanced diet rich in iron.",
            "Schedule regular health check-ups every six months.",
            "Stay hydrated and exercise regularly.",
            "Include a variety of fruits and vegetables in your diet.",
        ],
        "preventive_measures": [
            "Continue healthy eating habits.",
            "Include iron-rich foods in your weekly diet.",
            "Exercise regularly and get enough sleep.",
            "Stay informed about anemia symptoms.",
            "Consider annual blood tests for prevention.",
        ],
        "treatment_timeline": [
            {"phase": "Routine check-up", "duration": "Every 6 months"},
            {"phase": "Monitor for symptoms", "duration": "Ongoing"},
            {"phase": "Consult if symptoms appear", "duration": "As needed"},
        ],
    }


def get_cataract_recommendations(result):
    """Return guidance for the cataract result page."""
    if result == "Cataract":
        return {
            "severity": "Moderate",
            "detection_stage": "Early Detection",
            "precautions": [
                "Consult an ophthalmologist promptly.",
                "Wear UV protective sunglasses outdoors.",
                "Use proper lighting when reading.",
                "Follow the 20-20-20 rule to reduce eye strain.",
                "Use lubricating eye drops if dry eyes are a concern.",
            ],
            "preventive_measures": [
                "Avoid smoking because it increases cataract risk.",
                "Control blood sugar levels if you have diabetes.",
                "Eat antioxidant-rich foods such as vitamin C, vitamin E, and beta-carotene sources.",
                "Protect your eyes from UV radiation with sunglasses.",
                "Manage blood pressure with diet and exercise.",
            ],
            "treatment_timeline": [
                {"phase": "Ophthalmologist consultation", "duration": "Week 1"},
                {"phase": "Monitoring and prescription glasses", "duration": "Week 2-8"},
                {"phase": "Surgery consultation if vision worsens", "duration": "Month 2-3"},
                {"phase": "Post-surgery care if needed", "duration": "Month 3-4"},
            ],
        }

    return {
        "severity": "Normal",
        "detection_stage": "Healthy",
        "precautions": [
            "Schedule regular eye check-ups.",
            "Protect your eyes from UV light.",
            "Take breaks during screen time using the 20-20-20 rule.",
            "Maintain good eye hygiene.",
            "Blink regularly when using screens.",
        ],
        "preventive_measures": [
            "Eat eye-healthy foods such as carrots, leafy greens, fish, and eggs.",
            "Maintain a healthy lifestyle with regular exercise.",
            "Control diabetes and blood pressure if applicable.",
            "Wear protective eyewear when needed.",
            "Get annual comprehensive eye exams.",
        ],
        "treatment_timeline": [
            {"phase": "Routine eye exam", "duration": "Every year"},
            {"phase": "Monitor vision changes", "duration": "Ongoing"},
            {"phase": "Consult if any changes occur", "duration": "As needed"},
        ],
    }


@app.route("/check_models")
def check_models():
    """Return model-loading status and legacy artifact visibility."""
    global anemia_model, cataract_model
    ensure_models_loaded()

    anemia_path = resolve_first_existing(ANEMIA_MODEL_CANDIDATES)
    cataract_path = resolve_first_existing(CATARACT_MODEL_CANDIDATES)

    anemia_test = None
    cataract_test = None

    if anemia_model is not None:
        try:
            test_input = np.random.random((1, 224, 224, 3))
            test_input = tf.keras.applications.mobilenet_v3.preprocess_input(test_input)
            anemia_test = anemia_model.predict(test_input, verbose=0).tolist()
        except Exception as exc:
            anemia_test = f"Error during anemia smoke test: {exc}"

    if cataract_model is not None:
        try:
            test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            cataract_test = cataract_model.predict(test_input, verbose=0).tolist()
        except Exception as exc:
            cataract_test = f"Error during cataract smoke test: {exc}"

    status = {
        "anemia_model_candidates": [str(path) for path in ANEMIA_MODEL_CANDIDATES],
        "cataract_model_candidates": [str(path) for path in CATARACT_MODEL_CANDIDATES],
        "anemia_model_file_exists": anemia_path is not None,
        "cataract_model_file_exists": cataract_path is not None,
        "anemia_model_loaded": anemia_model is not None,
        "cataract_model_loaded": cataract_model is not None,
        "anemia_loaded_from": loaded_model_paths["anemia"],
        "cataract_loaded_from": loaded_model_paths["cataract"],
        "legacy_anemia_artifacts": list_existing_paths(ANEMIA_LEGACY_ARTIFACTS),
        "legacy_cataract_artifacts": list_existing_paths(CATARACT_LEGACY_ARTIFACTS),
        "models_folder_exists": MODELS_DIR.exists(),
        "models_folder_content": sorted(os.listdir(MODELS_DIR)) if MODELS_DIR.exists() else [],
        "current_directory": os.getcwd(),
        "anemia_test_prediction": anemia_test,
        "cataract_test_prediction": cataract_test,
        "anemia_class_names": ANEMIA_CLASS_NAMES,
        "cataract_class_names": CATARACT_CLASS_NAMES,
        "cataract_preprocessing": "RGB center-crop to square, resize to 224x224, EfficientNet preprocessing inside model",
    }
    return jsonify(status)


if __name__ == "__main__":
    init_db()
    anemia_loaded, cataract_loaded = load_models()

    print("\n" + "=" * 60)
    print("Flask application starting")
    print("=" * 60)
    print(f"Anemia model: {'LOADED' if anemia_loaded else 'NOT LOADED'}")
    print(f"Cataract model: {'LOADED' if cataract_loaded else 'NOT LOADED'}")
    print(f"Anemia class names: {ANEMIA_CLASS_NAMES}")
    print(f"Canonical anemia model: {loaded_model_paths['anemia']}")
    print(f"Canonical cataract model: {loaded_model_paths['cataract']}")
    print("Model status endpoint: http://localhost:5000/check_models")
    print("=" * 60 + "\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
