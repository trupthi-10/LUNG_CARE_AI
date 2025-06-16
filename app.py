from flask import Flask, render_template, request, redirect, session, flash
from database_config import init_mysql

app = Flask(__name__)
app.secret_key = 'your_secret_key'
mysql = init_mysql(app)

# ------------------ HOME PAGE ------------------
@app.route('/')
def home():
    return render_template('index.html')

# ------------------ USER & ADMIN REGISTRATION ------------------
import re
from flask import Flask, render_template, request, redirect, session, flash

@app.route('/register', methods=['GET', 'POST'])   
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        password = request.form['password']
        user_type = request.form['user_type']

        # ✅ Role check
        if user_type not in ['user', 'admin']:
            flash('Please select a valid role.', 'danger')
            return redirect('/register')

        # ✅ Name validation (letters and spaces only, min 2 characters)
        if not re.match(r'^[A-Za-z\s]{2,50}$', name):
            flash('Name should only contain letters and spaces.', 'danger')
            return redirect('/register')

        # ✅ Email validation (must be Gmail)
        email = request.form['email'].strip().lower()
        if not re.match(r'^[a-z][a-z0-9._%+-]*@gmail\.com$', email):
            flash('Only valid Gmail addresses are allowed.', 'danger')
            return redirect('/register')

        # ✅ Mobile validation (starts with 6-9, 10 digits only)
        if not re.match(r'^[6-9][0-9]{9}$', mobile):
            flash('Enter a valid 10-digit Indian mobile number.', 'danger')
            return redirect('/register')

        # ✅ Password validation (min 6, at least 1 uppercase and 1 special char)
        if not re.match(r'^(?=.*[A-Z])(?=.*[^A-Za-z0-9]).{6,}$', password):
            flash('Password must contain at least one uppercase letter, one special character, and be at least 6 characters long.', 'danger')
            return redirect('/register')

        # ✅ Check if email already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            cursor.close()
            flash('User already registered with this email.', 'warning')
            return redirect('/register')

        # ✅ Insert into DB
        cursor.execute("INSERT INTO users (name, email, mobile, password, user_type) VALUES (%s, %s, %s, %s, %s)",
                       (name, email, mobile, password, user_type))
        mysql.connection.commit()
        cursor.close()

        flash('Registered successfully! Please login.', 'success')
        return redirect('/login_user' if user_type == 'user' else '/login_admin')

    return render_template('register.html')

# ------------------ USER LOGIN ------------------
@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    error = None
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s AND user_type='user'", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            if user[4] == password:  # Adjust index if needed
                session['user_id'] = user[0]
                session['user_name'] = user[1]
                return redirect('/user_home')
            else:
                error = "Incorrect password. Please try again."
        else:
            error = "Email not found."

    return render_template('login_user.html', error=error)

# ------------------ ADMIN LOGIN ------------------
@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    error = None

    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s AND user_type = 'admin'", (email,))
        admin = cursor.fetchone()
        cursor.close()

        if admin:
            if admin[4] == password:  # 🔁 adjust if password isn't column 4
                session['admin_id'] = admin[0]
                session['admin_name'] = admin[1]
                return redirect('/admin_dashboard')
            else:
                error = "Incorrect password. Please try again."
        else:
            error = "Admin account not found."

    return render_template('login_admin.html', error=error)

#--------------------------------------------------------------
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        new_password = request.form['new_password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            cursor.execute("UPDATE users SET password = %s WHERE email = %s", (new_password, email))
            mysql.connection.commit()
            cursor.close()
            return redirect('/login_user')  # ✅ redirect works here
        else:
            cursor.close()
            return render_template('forgot_password.html', error="Email not found.")

    return render_template('forgot_password.html')

# ------------------ PLACEHOLDER DASHBOARDS ------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, session, render_template, request, redirect, send_file, url_for
from model import classify_patient_condition
from cxr_model import predict_cxr
from werkzeug.utils import secure_filename
from datetime import datetime
from fpdf import FPDF
from flask_mysqldb import MySQL

# Configure Upload and Report Paths
UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

@app.route('/user_home', methods=['GET', 'POST'])
def user_home():
    if 'user_id' not in session:
        return redirect('/login_user')

    name = session.get('user_name', 'User')
    user_id = session['user_id']
    cxr_result = vitals_result = confidence = gradcam_path = None
    graph_data = []
    latest_report = None
    report_message = ""
    vitals_chart_path = ""

    cursor = mysql.connection.cursor()

    # Get latest report
    cursor.execute("""
        SELECT disease, vital_trend, uploaded_at
        FROM diagnosis_results
        WHERE user_id = %s
        ORDER BY uploaded_at DESC
        LIMIT 1
    """, (user_id,))
    latest_report = cursor.fetchone()
    if not latest_report:
        report_message = "No previous report found."

    # Check user profile
    cursor.execute("SELECT mobile, age, sex, address FROM user_profile WHERE user_id = %s", (user_id,))
    profile = cursor.fetchone()

    if request.method == 'POST':
        # Get form values safely
        mobile = request.form.get('mobile')
        age = request.form.get('age')
        sex = request.form.get('sex')
        address = request.form.get('address')

        if not profile:
            # Insert new profile
            cursor.execute("""
                INSERT INTO user_profile (user_id, mobile, age, sex, address)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, mobile, age, sex, address))
            mysql.connection.commit()
        elif any(p is None for p in profile):
            # Update existing profile if incomplete
            cursor.execute("""
                UPDATE user_profile
                SET mobile = %s, age = %s, sex = %s, address = %s
                WHERE user_id = %s
            """, (mobile, age, sex, address, user_id))
            mysql.connection.commit()
        else:
            # Use existing values if profile is complete
            (mobile, age, sex, address) = profile

        now_str = datetime.now().strftime("%Y%m%d%H%M%S")

        # Process CXR
        if 'cxr' in request.files and request.files['cxr'].filename != '':
            cxr = request.files['cxr']
            cxr_filename = secure_filename(f"{now_str}_cxr_{cxr.filename}")
            cxr_path = os.path.join(UPLOAD_FOLDER, cxr_filename)
            cxr.save(cxr_path)
            cxr_result, confidence, gradcam_path = predict_cxr(cxr_path)

        # Process Vitals
        if 'vitals' in request.files and request.files['vitals'].filename != '':
            vitals = request.files['vitals']
            vitals_filename = secure_filename(f"{now_str}_vitals_{vitals.filename}")
            vitals_path = os.path.join(UPLOAD_FOLDER, vitals_filename)
            vitals.save(vitals_path)
            vitals_result = classify_patient_condition(vitals_path)

            # Generate vitals trend chart
            df = pd.read_csv(vitals_path)
            df.columns = [col.strip().lower() for col in df.columns]
            df.rename(columns={'heart rate (bpm)': 'heart_rate', 'blood oxygen level (%)': 'spo2'}, inplace=True)

            if 'heart_rate' in df.columns and 'spo2' in df.columns:
                df = df[['heart_rate', 'spo2']].dropna()
                graph_data = df.to_dict(orient='records')

                vitals_chart_path = os.path.join(REPORT_FOLDER, f"vitals_chart_{user_id}_{now_str}.png")
                plt.figure(figsize=(5, 3))
                plt.plot(df['heart_rate'], label='Heart Rate (bpm)', color='red', marker='o')
                plt.plot(df['spo2'], label='SpO2 (%)', color='blue', marker='o')
                plt.title("Vitals Trend Over Time")
                plt.xlabel("Time Index")
                plt.ylabel("Value")
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(vitals_chart_path)
                plt.close()

        # Save diagnosis if results exist
        if cxr_result or vitals_result:
            cursor.execute("""
                INSERT INTO diagnosis_results 
                (user_id, disease, vital_trend, mobile, age, sex, address, uploaded_at) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (user_id, cxr_result, vitals_result, mobile, age, sex, address))
            mysql.connection.commit()

            # Generate PDF Report
            report_filename = f"report_{user_id}_{now_str}.pdf"
            report_path = os.path.join(REPORT_FOLDER, report_filename)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "LungCare AI - Diagnosis Report", ln=True, align='C')
            pdf.set_font("Arial", "", 12)
            pdf.ln(10)

            pdf.cell(0, 10, f"Name: {name}", ln=True)
            pdf.cell(0, 10, f"Mobile: {mobile}", ln=True)
            pdf.cell(0, 10, f"Age: {age}", ln=True)
            pdf.cell(0, 10, f"Sex: {sex}", ln=True)
            pdf.multi_cell(0, 10, f"Address: {address}")
            pdf.ln(5)

            if cxr_result:
                pdf.cell(0, 10, f"Disease Detected: {cxr_result}", ln=True)
                pdf.cell(0, 10, f"Confidence: {round(confidence * 100, 2)}%", ln=True)

            if vitals_result:
                pdf.cell(0, 10, f"Vitals Status: {vitals_result}", ln=True)

            if gradcam_path and os.path.exists(gradcam_path):
                try:
                    pdf.image(gradcam_path, x=60, y=pdf.get_y() + 10, w=80)
                    pdf.ln(50)
                except:
                    pass

            if vitals_chart_path and os.path.exists(vitals_chart_path):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Vitals Visualization", ln=True)
                pdf.image(vitals_chart_path, x=30, y=pdf.get_y() + 5, w=150)

            pdf.output(report_path)
            session['latest_report_file'] = report_filename
            latest_report = (cxr_result or "-", vitals_result or "-", "Just now")
            report_message = ""

    
    # Fetch profile from DB
    cursor.execute("SELECT mobile, age, sex, address FROM user_profile WHERE user_id = %s", (user_id,))
    profile = cursor.fetchone()

    if profile and all(profile):  # Checks all fields are non-null
        profile_complete = True
    else:
        profile_complete = False

    profile_complete = bool(profile and all(p is not None for p in profile))
    # Unpack individual profile fields for template use
    mobile = profile[0] if profile else None
    age = profile[1] if profile else None
    sex = profile[2] if profile else None
    address = profile[3] if profile else None

    cursor.close()
    return render_template(
    "user_home.html",
    name=name,
    prediction=cxr_result,
    confidence=confidence,
    gradcam=gradcam_path,
    vitals=vitals_result,
    graph=graph_data,
    latest_report=latest_report,
    report_message=report_message,
    profile_complete=profile_complete,
    mobile=mobile,         
    age=age,               
    sex=sex,               
    address=address        
)

@app.route('/download_report')
def download_report():
    filename = request.args.get('file') or session.get('latest_report_file')
    if filename:
        path = os.path.join(REPORT_FOLDER, filename)
        if os.path.exists(path):
            return send_file(path, as_attachment=True)
    return "Report not found", 404

#-----------------------------------------------------------------------------------------
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'admin_id' not in session:
        return redirect('/login_admin')

    cursor = mysql.connection.cursor()
    
    # Get all users
    cursor.execute("SELECT id, name, email, mobile FROM users WHERE user_type='user'")
    users = cursor.fetchall()

    # Get all diagnosis results
    cursor.execute("""
        SELECT dr.user_id, u.name, dr.disease, dr.vital_trend, dr.uploaded_at
        FROM diagnosis_results dr
        JOIN users u ON dr.user_id = u.id
        ORDER BY dr.uploaded_at DESC
    """)
    reports = cursor.fetchall()
    
    cursor.close()

    return render_template('admin_dashboard.html', users=users, reports=reports)


# ------------------ LOGOUT ------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
