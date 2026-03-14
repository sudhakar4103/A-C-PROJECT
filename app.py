import os
import sqlite3
import numpy as np
import cv2
import pickle
from flask import Flask, render_template, request, redirect, url_for, session, flash
import matplotlib.image as mpimg



# ================== Flask App ==================
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # required for session management
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ================== Load Data ==================
with open('dot.pickle', 'rb') as f:
    dot1 = pickle.load(f)

with open('labels.pickle', 'rb') as f:
    labels1 = pickle.load(f)

classes = ['Normal','Cataract','Anemia','NoAnemia']

# ================== SQLite Setup ==================
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                      )''')
    conn.commit()
    conn.close()

init_db()

# ================== Routes ==================

# ---- Registration ----
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        conn.close()

    return render_template('register.html')


# ---- Login ----
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials!", "danger")
            return redirect(url_for('login'))

    return render_template('index.html')


# ---- Logout ----
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('login'))

# ---- Prediction Page ----
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected!", "danger")
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = mpimg.imread(filepath)
        img = cv2.resize(img, (50,50))

        try:
            gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            gray1 = img

        class_names = {
            0: 'Normal',
            1: 'Cataract',
            2: 'Anemia',
            3: 'NoAnemia',
        }

        precautions_data = {
            "Anemia": {
                "title": "Anemia Management",
                "precautions": ["Eat iron-rich foods", "Take Vitamin C"],
                "preventions": ["Balanced diet", "Medical checkups"]
            },
            "Cataract": {
                "title": "Cataract Care",
                "precautions": ["Wear UV sunglasses", "Regular eye exams"],
                "preventions": ["Quit smoking", "Manage blood sugar"]
            }
        }

        # ✅ REQUIRED FIX
        a = "Unknown"
        advice = None

        temp_data1 = []
        for ijk in range(len(dot1)):
            temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
            temp_data1.append(temp_data)

        temp_data1 = np.array(temp_data1)
        zz = np.where(temp_data1 == 1)

        if zz[0].size > 0:
            identified_class = labels1[zz[0][0]]
            if identified_class in class_names:
                a = class_names[identified_class]
                advice = precautions_data.get(a, None)

        return render_template(
            'result.html',
            blood_group=a,
            filename=file.filename,
            advice=advice
        )

    return render_template('predict.html')




# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if 'username' not in session:
#         flash("Please login first.", "warning")
#         return redirect(url_for('login'))

#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash("No file uploaded!", "danger")
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             flash("No file selected!", "danger")
#             return redirect(request.url)

#         # Save uploaded file
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         # Read image
#         img = mpimg.imread(filepath)
#         img = cv2.resize(img, (50,50))

#         # Convert to grayscale
#         try:
#             gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         except:
#             gray1 = img



#         class_names = {
#             0: 'Normal',
#             1: 'Cataract',
#             2: 'Anemia',
#             3: 'NoAnemia',
    
#         }
        
#         precautions_data = {
#                 "Anemia": {
#                     "title": "Anemia Management",
#                     "precautions": ["Eat iron-rich foods", "Take Vitamin C"],
#                     "preventions": ["Balanced diet", "Medical checkups"]
#                 },
#                 "Cataract": {
#                     "title": "Cataract Care",
#                     "precautions": ["Wear UV sunglasses", "Regular eye exams"],
#                     "preventions": ["Quit smoking", "Manage blood sugar"]
#                 }
#             }
        
        
        
#         temp_data1 = []
#         for ijk in range(0, len(dot1)):
#             temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))  # Comparing with gray1
#             temp_data1.append(temp_data)


#         temp_data1 = np.array(temp_data1)

#         zz = np.where(temp_data1 == 1)

#         if zz[0].size > 0: 
#             identified_class = labels1[zz[0][0]]  
#             print("----------------------------------------")
#             if identified_class in class_names:
#                 print(f"Identified as {class_names[identified_class]}")
#                 a=class_names[identified_class]
#                 print(a)
#                 advice = precautions_data.get(a, None)
#             else:
#                 print("Class not recognized.")
#                 a="No"
#                 print(a)
#             print("----------------------------------------")
        
        

#         return render_template('result.html', blood_group=a, filename=file.filename,advice=advice)

#     return render_template('predict.html')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, use_reloader=False)