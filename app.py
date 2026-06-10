import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time

st.set_page_config(page_title="Career Path Predictor", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');

    .stApp { background-color: #f5f0e8; }

    .hero {
        background: linear-gradient(135deg, #4a3728 0%, #8b6f47 50%, #c4956a 100%);
        padding: 50px 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(74,55,40,0.3);
    }
    .hero-title {
        font-size: 52px;
        font-weight: bold;
        color: white;
        font-family: 'Playfair Display', serif;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        font-size: 18px;
        color: #f5f0e8;
        opacity: 0.9;
    }
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(74,55,40,0.1);
        border-bottom: 4px solid #8b6f47;
        transition: transform 0.2s;
    }
    .stat-number {
        font-size: 36px;
        font-weight: bold;
        color: #8b6f47;
    }
    .stat-label {
        font-size: 13px;
        color: #7a6652;
        margin-top: 5px;
    }
    .step-header {
        background: linear-gradient(90deg, #8b6f47, #c4956a);
        padding: 12px 20px;
        border-radius: 10px;
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .result-box {
        background: linear-gradient(135deg, #4a3728, #8b6f47);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(74,55,40,0.4);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 8px 25px rgba(74,55,40,0.4); }
        50% { box-shadow: 0 8px 35px rgba(74,55,40,0.7); }
        100% { box-shadow: 0 8px 25px rgba(74,55,40,0.4); }
    }
    .career-card-gold {
        background: linear-gradient(135deg, #fff9f0, #fff3e0);
        padding: 20px 25px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 6px solid #f9a825;
        box-shadow: 0 4px 12px rgba(74,55,40,0.1);
        color: #4a3728;
    }
    .career-card-silver {
        background: linear-gradient(135deg, #f8f8f8, #efefef);
        padding: 20px 25px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 6px solid #9e9e9e;
        box-shadow: 0 4px 12px rgba(74,55,40,0.08);
        color: #4a3728;
    }
    .career-card-bronze {
        background: linear-gradient(135deg, #fff5ec, #ffe8d6);
        padding: 20px 25px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 6px solid #c4956a;
        box-shadow: 0 4px 12px rgba(74,55,40,0.08);
        color: #4a3728;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: none;
        margin: 10px 0;
        color: #4a3728;
        box-shadow: 0 4px 12px rgba(74,55,40,0.08);
    }
    .basis-card {
        background: white;
        padding: 14px 20px;
        border-radius: 10px;
        border-left: 4px solid #c4956a;
        color: #4a3728;
        margin: 6px 0;
        box-shadow: 0 2px 8px rgba(74,55,40,0.06);
    }
    .suggestion-box {
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        padding: 18px 20px;
        border-radius: 12px;
        border-left: 5px solid #43a047;
        color: #1b5e20;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(67,160,71,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1, #fffde7);
        padding: 18px 20px;
        border-radius: 12px;
        border-left: 5px solid #f9a825;
        color: #4a3728;
        margin: 10px 0;
        box-shadow: 0 3px 10px rgba(249,168,37,0.1);
    }
    .roadmap-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(74,55,40,0.1);
        border-top: 4px solid #8b6f47;
        color: #4a3728;
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #8b6f47, #c4956a, #f5f0e8);
        border-radius: 2px;
        margin: 25px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4a3728, #8b6f47, #c4956a);
        color: white;
        border: none;
        padding: 15px 50px;
        border-radius: 30px;
        font-size: 20px;
        font-weight: bold;
        width: 100%;
        box-shadow: 0 6px 20px rgba(74,55,40,0.4);
        transition: all 0.3s;
    }
    h1, h2, h3, h4 { color: #4a3728 !important; }
    p { color: #4a3728; }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #4a3728 !important;
        font-weight: 600;
        font-size: 15px;
    }

    /* =============================================
       FIX 1: DROPDOWN / SELECTBOX TEXT VISIBILITY
       ============================================= */
    /* The select box container and selected text */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] > div {
        background-color: white !important;
        color: #4a3728 !important;
    }
    /* Every text/span inside select */
    .stSelectbox [data-baseweb="select"] span,
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] p,
    .stSelectbox [data-baseweb="select"] input {
        color: #4a3728 !important;
        background-color: white !important;
    }
    /* The open dropdown list */
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] * {
        background-color: white !important;
        color: #4a3728 !important;
    }
    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] li:hover {
        background-color: #f5f0e8 !important;
        color: #4a3728 !important;
    }
    /* Fallback for any stSelectbox testid */
    [data-testid="stSelectboxContainer"],
    [data-testid="stSelectboxContainer"] * {
        color: #4a3728 !important;
        background-color: white !important;
    }
    /* =============================================
       FIX 2: PLOTLY CHART TEXT VISIBILITY
       ============================================= */
    .js-plotly-plot .plotly text {
        fill: #4a3728 !important;
    }
    .js-plotly-plot .plotly .gtitle text,
    .js-plotly-plot .plotly .xtitle text,
    .js-plotly-plot .plotly .ytitle text {
        fill: #4a3728 !important;
    }
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #4a3728 !important;
    }
    .js-plotly-plot .plotly .legendtext {
        fill: #4a3728 !important;
    }
    .js-plotly-plot .plotly .annotation-text {
        fill: #4a3728 !important;
    }
    /* ============================================= */

    .progress-step {
        background: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin: 0 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA ====================
skills_map = {
    'Science (Matric)': ['Mathematics', 'Basic Computer', 'Communication', 'English', 'Lab Skills'],
    'Arts (Matric)': ['Urdu Literature', 'Communication', 'English', 'Drawing', 'Creative Writing'],
    'Commerce (Matric)': ['Basic Accounting', 'Math', 'MS Office', 'English', 'Communication'],
    'ICS': ['Python', 'C++', 'Web Development', 'Cybersecurity', 'Networking', 'Database Management', 'AI/ML'],
    'Pre-Engineering': ['AutoCAD', 'Mathematics', 'Physics', 'Problem Solving', 'MATLAB', 'Circuit Design'],
    'Pre-Medical': ['Biology', 'Chemistry', 'Lab Skills', 'Patient Care', 'Anatomy', 'First Aid'],
    'Commerce (Inter)': ['Accounting', 'Tally ERP', 'MS Excel', 'Business Communication', 'Finance', 'Taxation'],
    'Arts (Inter)': ['Content Writing', 'Graphic Design', 'Urdu Literature', 'Fine Arts', 'Media', 'Communication'],
    'CS/IT': ['Python', 'Java', 'Web Development', 'AI/ML', 'Cybersecurity', 'Cloud Computing', 'Data Science', 'DevOps'],
    'Cybersecurity (BS)': ['Network Security', 'Ethical Hacking', 'Penetration Testing', 'Cryptography', 'Incident Response', 'Malware Analysis', 'SIEM Tools', 'Cloud Security'],
    'MS Cybersecurity': ['Advanced Penetration Testing', 'Digital Forensics', 'Threat Intelligence', 'Security Architecture', 'Zero Trust', 'Malware Reverse Engineering'],
    'Undergraduate CS/IT': ['Python', 'Java', 'Web Development', 'AI/ML', 'Cybersecurity', 'Cloud Computing', 'Data Science', 'Freelance Gigs'],
    'Undergraduate Cybersecurity': ['Network Security', 'Ethical Hacking', 'Penetration Testing', 'Cryptography', 'Bug Bounty'],
    'Undergraduate Engineering': ['AutoCAD', 'MATLAB', 'Circuit Design', 'Civil Design', 'Mechanical Design', 'Problem Solving'],
    'Undergraduate Business': ['Financial Analysis', 'Marketing', 'Accounting', 'E-commerce', 'Business Strategy'],
    'Undergraduate Arts/Media': ['Content Writing', 'Graphic Design', 'Journalism', 'Fine Arts', 'Media Production'],
    'Undergraduate Medical': ['Biology', 'Clinical Skills', 'Patient Care', 'Lab Analysis', 'Medical Writing'],
    'Engineering': ['AutoCAD', 'MATLAB', 'Project Management', 'Circuit Design', 'Civil Design', 'Mechanical Design'],
    'Medical/Health': ['Clinical Skills', 'Patient Care', 'Surgery Basics', 'Pharmacology', 'Lab Analysis'],
    'Business/Commerce': ['Financial Analysis', 'Marketing', 'HR Management', 'Business Strategy', 'Accounting', 'Supply Chain', 'E-commerce'],
    'Arts/Humanities': ['Content Writing', 'Journalism', 'Graphic Design', 'Fine Arts', 'Media Production', 'Photography'],
    'Law': ['Legal Research', 'Case Analysis', 'Contract Law', 'Criminal Law', 'Corporate Law'],
    'Education': ['Teaching', 'Curriculum Design', 'Communication', 'Child Psychology', 'E-Learning'],
    'MS/MPhil CS': ['Advanced AI', 'Research', 'Deep Learning', 'Big Data', 'Cloud Architecture', 'NLP'],
    'MBA': ['Leadership', 'Strategic Management', 'Finance', 'Marketing Analytics', 'Entrepreneurship', 'Consulting'],
    'MS Engineering': ['Advanced CAD', 'Research', 'Simulation', 'Project Management', 'Systems Design'],
    'MS Medical': ['Advanced Clinical', 'Research', 'Specialization', 'Medical Writing', 'Public Health'],
    'MA/MSc Arts': ['Advanced Research', 'Academic Writing', 'Teaching', 'Cultural Studies', 'Policy Analysis'],
}

cert_map = {
    'Science (Matric)': ['None'], 'Arts (Matric)': ['None'], 'Commerce (Matric)': ['None'],
    'ICS': ['None', 'Cisco CCNA', 'Google IT Support', 'AWS Cloud Practitioner', 'CompTIA Security+'],
    'Pre-Engineering': ['None', 'AutoCAD Certified', 'PMP', 'Six Sigma'],
    'Pre-Medical': ['None', 'First Aid', 'CPR Certified', 'Lab Safety'],
    'Commerce (Inter)': ['None', 'Tally ERP', 'QuickBooks', 'Google Digital Marketing'],
    'Arts (Inter)': ['None', 'Adobe Certified', 'Google Analytics', 'Content Marketing'],
    'CS/IT': ['None', 'AWS Certified', 'Google Cloud', 'Microsoft Azure', 'Cisco CCNA', 'CompTIA Security+'],
    'Cybersecurity (BS)': ['None', 'CompTIA Security+', 'CEH (Certified Ethical Hacker)', 'Cisco CyberOps', 'OSCP', 'CompTIA CySA+'],
    'MS Cybersecurity': ['None', 'CISSP', 'OSCP', 'CEH Advanced', 'CISM', 'CompTIA CASP+'],
    'Undergraduate CS/IT': ['None', 'Google IT Support', 'AWS Cloud Practitioner', 'Meta Front-End Developer', 'CompTIA ITF+'],
    'Undergraduate Cybersecurity': ['None', 'CompTIA Security+', 'CEH', 'Cisco CyberOps'],
    'Undergraduate Engineering': ['None', 'AutoCAD Certified', 'PMP Foundation', 'Six Sigma Yellow Belt'],
    'Undergraduate Business': ['None', 'Google Digital Marketing', 'HubSpot Marketing', 'QuickBooks'],
    'Undergraduate Arts/Media': ['None', 'Adobe Certified', 'Google Analytics', 'HubSpot Content'],
    'Undergraduate Medical': ['None', 'First Aid', 'CPR Certified', 'Lab Safety'],
    'Engineering': ['None', 'PMP', 'AutoCAD Certified', 'Six Sigma'],
    'Medical/Health': ['None', 'USMLE', 'BLS Certified', 'ACLS'],
    'Business/Commerce': ['None', 'CPA', 'CFA', 'Google Analytics', 'HubSpot Marketing', 'PMP'],
    'Arts/Humanities': ['None', 'Adobe Certified', 'Google Analytics', 'HubSpot Content'],
    'Law': ['None', 'Bar Certified', 'Legal Research Certificate'],
    'Education': ['None', 'Teaching Certificate', 'Cambridge CELTA'],
    'MS/MPhil CS': ['None', 'AWS Solutions Architect', 'Google Cloud Professional', 'TensorFlow Developer'],
    'MBA': ['None', 'CFA', 'PMP', 'Six Sigma Black Belt'],
    'MS Engineering': ['None', 'PMP', 'Six Sigma', 'PE License'],
    'MS Medical': ['None', 'Board Certification', 'Research Ethics'],
    'MA/MSc Arts': ['None', 'Research Certification', 'TEFL'],
}

career_info = {
    'Software Engineer': {'emoji': '💻', 'desc': 'Design, develop and maintain software applications.', 'salary': 'PKR 80,000 - 300,000/month', 'degree': "Bachelor's in CS/IT", 'scope': 'Very High 🚀', 'areas': 'Karachi, Lahore, Islamabad, Remote (Global)', 'growth': 85},
    'ML Engineer': {'emoji': '🤖', 'desc': 'Build and deploy machine learning models and AI systems.', 'salary': 'PKR 120,000 - 400,000/month', 'degree': "Bachelor's/Master's in CS", 'scope': 'Extremely High 🔥', 'areas': 'Karachi, Lahore, Islamabad, Remote (Global)', 'growth': 95},
    'Cybersecurity Expert': {'emoji': '🔐', 'desc': 'Protect computer systems from cyber attacks.', 'salary': 'PKR 100,000 - 350,000/month', 'degree': "Bachelor's in CS/IT", 'scope': 'Very High 🚀', 'areas': 'Karachi, Lahore, Islamabad, Remote', 'growth': 90},
    'Doctor': {'emoji': '👨‍⚕️', 'desc': 'Diagnose and treat patients, promote health.', 'salary': 'PKR 100,000 - 500,000/month', 'degree': 'MBBS + specialization', 'scope': 'Always High ❤️', 'areas': 'All cities, Middle East', 'growth': 80},
    'Business Analyst': {'emoji': '📊', 'desc': 'Analyze business processes and recommend improvements.', 'salary': 'PKR 70,000 - 250,000/month', 'degree': "Bachelor's in Business", 'scope': 'High 📈', 'areas': 'Karachi, Lahore, Islamabad', 'growth': 75},
    'Financial Analyst': {'emoji': '💰', 'desc': 'Analyze financial data to guide investment decisions.', 'salary': 'PKR 80,000 - 280,000/month', 'degree': "Bachelor's in Finance", 'scope': 'High 📈', 'areas': 'Karachi, Lahore, Islamabad', 'growth': 72},
    'Marketing Executive': {'emoji': '📣', 'desc': 'Plan and execute marketing campaigns.', 'salary': 'PKR 50,000 - 150,000/month', 'degree': "Bachelor's in Marketing", 'scope': 'High 📈', 'areas': 'Karachi, Lahore, Islamabad, Remote', 'growth': 70},
    'Lawyer': {'emoji': '⚖️', 'desc': 'Represent clients in legal matters.', 'salary': 'PKR 60,000 - 300,000/month', 'degree': "LLB / Bachelor's in Law", 'scope': 'Stable ✅', 'areas': 'All cities, All courts', 'growth': 65},
    'Teacher/Professor': {'emoji': '👩‍🏫', 'desc': 'Educate students and conduct research.', 'salary': 'PKR 40,000 - 250,000/month', 'degree': "Bachelor's/Master's", 'scope': 'Stable ✅', 'areas': 'All cities', 'growth': 60},
    'Graphic Designer': {'emoji': '🎨', 'desc': 'Create visual content for brands and media.', 'salary': 'PKR 40,000 - 150,000/month', 'degree': "Bachelor's in Fine Arts", 'scope': 'Growing 🌱', 'areas': 'Karachi, Lahore, Remote (Freelancing)', 'growth': 68},
    'Content Writer': {'emoji': '✍️', 'desc': 'Create written content for digital platforms.', 'salary': 'PKR 30,000 - 120,000/month', 'degree': "Bachelor's in Arts/Journalism", 'scope': 'Growing 🌱', 'areas': 'Remote (Global Freelancing)', 'growth': 65},
    'Accountant': {'emoji': '🧾', 'desc': 'Maintain financial records and handle taxes.', 'salary': 'PKR 40,000 - 150,000/month', 'degree': "Bachelor's in Accounting", 'scope': 'Stable ✅', 'areas': 'All major cities', 'growth': 62},
}

edu_groups = {
    'Matric': ['Science (Matric)', 'Arts (Matric)', 'Commerce (Matric)'],
    'Intermediate': ['ICS', 'Pre-Engineering', 'Pre-Medical', 'Commerce (Inter)', 'Arts (Inter)'],
    'Undergraduate (Current Student)': ['Undergraduate CS/IT', 'Undergraduate Cybersecurity', 'Undergraduate Engineering', 'Undergraduate Business', 'Undergraduate Arts/Media', 'Undergraduate Medical'],
    "Bachelor's (Completed)": ['CS/IT', 'Cybersecurity (BS)', 'Engineering', 'Medical/Health', 'Business/Commerce', 'Arts/Humanities', 'Law', 'Education'],
    "Master's": ['MS/MPhil CS', 'MS Cybersecurity', 'MBA', 'MS Engineering', 'MS Medical', 'MA/MSc Arts'],
}

edu_suggestions = {
    'ICS': {'degree': "Bachelor's in CS/IT or Software Engineering", 'careers': ['Software Engineer', 'Cybersecurity Expert', 'ML Engineer'], 'freelance': ['Web Development', 'Python', 'Cybersecurity']},
    'Pre-Engineering': {'degree': "Bachelor's in Engineering", 'careers': ['Civil/Electrical/Mechanical Engineer'], 'freelance': ['AutoCAD Design', '3D Modeling']},
    'Pre-Medical': {'degree': "MBBS or Health Sciences", 'careers': ['Doctor', 'Pharmacist', 'Lab Technician'], 'freelance': ['Medical Writing']},
    'Commerce (Inter)': {'degree': "Bachelor's in Commerce or BBA", 'careers': ['Accountant', 'Financial Analyst'], 'freelance': ['Bookkeeping', 'Virtual Assistant']},
    'Arts (Inter)': {'degree': "Bachelor's in Arts or Media", 'careers': ['Content Writer', 'Graphic Designer'], 'freelance': ['Graphic Design', 'Content Writing']},
    'Science (Matric)': {'degree': "Intermediate in ICS or Pre-Engineering", 'careers': ['Software Engineer or Engineer (after study)'], 'freelance': ['Learn a skill first']},
    'Arts (Matric)': {'degree': "Intermediate in Arts", 'careers': ['Content Writer or Teacher (after study)'], 'freelance': ['Learn a skill first']},
    'Commerce (Matric)': {'degree': "Intermediate in Commerce", 'careers': ['Accountant (after study)'], 'freelance': ['Learn a skill first']},
}

@st.cache_data
def load_and_train():
    df = pd.read_csv('career_data.csv')
    le_edu = LabelEncoder()
    le_group = LabelEncoder()
    le_skill = LabelEncoder()
    le_interest = LabelEncoder()
    le_work = LabelEncoder()
    le_career = LabelEncoder()
    df['edu_enc'] = le_edu.fit_transform(df['education'])
    df['group_enc'] = le_group.fit_transform(df['group'])
    df['skill_enc'] = le_skill.fit_transform(df['skill'])
    df['interest_enc'] = le_interest.fit_transform(df['interest_area'])
    df['work_enc'] = le_work.fit_transform(df['work_preference'])
    df['career_enc'] = le_career.fit_transform(df['career'])
    X = df[['edu_enc', 'group_enc', 'skill_enc', 'interest_enc', 'work_enc', 'cgpa']]
    y = df['career_enc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le_edu, le_group, le_skill, le_interest, le_work, le_career, df, acc

model, le_edu, le_group, le_skill, le_interest, le_work, le_career, df, acc = load_and_train()

# ==================== HERO ====================
st.markdown("""
<div class="hero">
    <div class="hero-title">🎯 Career Path Predictor</div>
    <div class="hero-subtitle">AI-powered career guidance for Pakistani students — Find your perfect career path</div>
</div>
""", unsafe_allow_html=True)

# Stats
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-number">721</div><div class="stat-label">📚 Students in Dataset</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-number">{acc*100:.0f}%</div><div class="stat-label">🎯 Model Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-number">12</div><div class="stat-label">💼 Career Paths</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-number">4</div><div class="stat-label">🎓 Education Levels</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Progress indicator
st.markdown("""
<div style='text-align:center; margin: 15px 0;'>
    <span style='background:#8b6f47; color:white; padding:8px 20px; border-radius:20px; margin:5px; font-weight:bold;'>1️⃣ Education</span>
    <span style='color:#8b6f47; font-size:20px;'>→</span>
    <span style='background:#c4956a; color:white; padding:8px 20px; border-radius:20px; margin:5px; font-weight:bold;'>2️⃣ Skills</span>
    <span style='color:#8b6f47; font-size:20px;'>→</span>
    <span style='background:#d4a574; color:white; padding:8px 20px; border-radius:20px; margin:5px; font-weight:bold;'>3️⃣ Preferences</span>
    <span style='color:#8b6f47; font-size:20px;'>→</span>
    <span style='background:#e8c9a0; color:#4a3728; padding:8px 20px; border-radius:20px; margin:5px; font-weight:bold;'>4️⃣ Results</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ==================== STEP 1 ====================
st.markdown('<div class="step-header">📚 Step 1 — Education Background</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    education = st.selectbox("🎓 Education Level", list(edu_groups.keys()))
with col2:
    group = st.selectbox("📖 Your Group/Field", edu_groups[education])

gpa = 3.0
percentage = 75
if education in ['Matric', 'Intermediate']:
    percentage = st.slider("📊 Percentage (%)", 33, 100, 75)
    cgpa = percentage / 25
    st.markdown(f"<p style='color:#8b6f47; font-size:13px;'>💡 Your percentage: <b>{percentage}%</b></p>", unsafe_allow_html=True)
elif education == 'Undergraduate (Current Student)':
    semester = st.selectbox("📅 Current Semester", ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'])
    gpa = st.slider("📊 Current CGPA (out of 4.0)", 0.0, 4.0, 2.8, step=0.1)
    cgpa = gpa * 25
    gpa_label = "Excellent 🌟" if gpa >= 3.5 else "Good 👍" if gpa >= 3.0 else "Average 📚" if gpa >= 2.5 else "Needs Improvement ⚠️"
    st.markdown(f"<p style='color:#8b6f47; font-size:13px;'>💡 Semester: <b>{semester}</b> | CGPA: <b>{gpa}/4.0</b> — {gpa_label}</p>", unsafe_allow_html=True)
else:
    gpa = st.slider("📊 GPA (out of 4.0)", 0.0, 4.0, 3.0, step=0.1)
    cgpa = gpa * 25
    gpa_label = "Excellent 🌟" if gpa >= 3.5 else "Good 👍" if gpa >= 3.0 else "Average 📚" if gpa >= 2.5 else "Needs Improvement ⚠️"
    st.markdown(f"<p style='color:#8b6f47; font-size:13px;'>💡 Your GPA: <b>{gpa}/4.0</b> — {gpa_label}</p>", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ==================== STEP 2 ====================
st.markdown('<div class="step-header">💡 Step 2 — Skills & Certifications</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    skill = st.selectbox("🛠️ Primary Skill", skills_map.get(group, ['Communication']))
with col4:
    cert = st.selectbox("📜 Certification (if any)", cert_map.get(group, ['None']))

if cert != 'None':
    st.markdown(f'<div class="suggestion-box">🏆 <b>Great!</b> Having a <b>{cert}</b> certification significantly boosts your career prospects!</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ==================== STEP 3 ====================
if education in ['Matric', 'Intermediate']:
    st.markdown('<div class="step-header">🌟 Step 3 — Your Current Situation</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        has_laptop = st.radio("💻 Do you have Laptop/PC?", ["Yes", "No"])
        internet = st.radio("🌐 Internet Access", ["Good", "Average", "Poor"])
    with col6:
        city_type = st.radio("🏙️ City Type", ["Big City", "Small City", "Village/Rural Area"])
        relocate = st.radio("🚀 Willing to Relocate?", ["Yes", "No"])
    work_pref = "Any"
    interest_area = "Technology"
    english_level = "Basic"
    job_priority = "Quick Employment"
    go_abroad = "No"
    gpa_trend = "Stable"
    extracurricular = "None"
elif education == 'Undergraduate (Current Student)':
    st.markdown('<div class="step-header">🌟 Step 3 — Freelancing & Situation</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        has_laptop = st.radio("💻 Do you have Laptop/PC?", ["Yes", "No"])
        internet = st.radio("🌐 Internet Access", ["Good", "Average", "Poor"])
        english_level = st.radio("🗣️ English Proficiency", ["Basic", "Intermediate", "Fluent"])
    with col6:
        city_type = st.radio("🏙️ City Type", ["Big City", "Small City", "Village/Rural Area"])
        gpa_trend = st.radio("📈 Academic Trend", ["Improving", "Stable", "Declining"])
        freelance_time = st.radio("⏰ Free Time for Freelancing", ["A lot (10+ hrs/week)", "Some (5-10 hrs/week)", "Very Less (<5 hrs/week)"])
    work_pref = "Remote"
    interest_area = "Technology"
    job_priority = "Freelancing"
    go_abroad = "No"
    extracurricular = "None"
    relocate = "No"
else:
    st.markdown('<div class="step-header">🌟 Step 3 — Preferences & Personal Situation</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        work_pref = st.radio("💼 Work Preference", ["Office", "Remote", "Fieldwork", "Any"])
        interest_area = st.radio("🌟 Interest Area", ["Technology", "Business", "Healthcare", "Arts & Media", "Education & Research"])
        english_level = st.radio("🗣️ English Proficiency", ["Basic", "Intermediate", "Fluent"])
    with col6:
        city_type = st.radio("🏙️ City Type", ["Big City (Karachi/Lahore/Islamabad)", "Small City", "Village/Rural Area"])
        job_priority = st.radio("🎯 Job Priority", ["High Salary", "Job Stability", "Passion/Interest", "Quick Employment"])
        relocate = st.radio("🚀 Willing to Relocate?", ["Yes", "No"])

    col7, col8 = st.columns(2)
    with col7:
        go_abroad = st.radio("✈️ Interested in Going Abroad?", ["Yes", "No"])
        has_laptop = st.radio("💻 Have Laptop/PC?", ["Yes", "No"])
    with col8:
        internet = st.radio("🌐 Internet Access", ["Good", "Average", "Poor"])
        gpa_trend = st.radio("📈 Academic Trend", ["Improving", "Stable", "Declining"])
        extracurricular = st.radio("🏆 Extra Curricular", ["Sports", "Debates/Public Speaking", "Volunteering", "Arts/Creative", "None"])

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ==================== PREDICT BUTTON ====================
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    predict_btn = st.button("🎯 Predict My Career Path!")

# ==================== RESULTS ====================
if predict_btn:

    with st.spinner("🤖 AI is analyzing your profile..."):
        time.sleep(1.5)

    # ===== UNDERGRADUATE =====
    if education == 'Undergraduate (Current Student)':
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## Undergraduate Freelancing & Career Guide")

        sem_num = int(semester[0]) if semester and semester[0].isdigit() else 1

        if gpa >= 3.5:
            cgpa_msg = 'Excellent CGPA! You can comfortably manage studies + freelancing.'
            cgpa_color = '#e8f5e9'; cgpa_border = '#43a047'
        elif gpa >= 3.0:
            cgpa_msg = 'Good CGPA. You can manage studies + freelancing with proper time management.'
            cgpa_color = '#e8f5e9'; cgpa_border = '#43a047'
        elif gpa >= 2.5:
            cgpa_msg = 'Average CGPA. Improve grades first. Only light freelancing (5-8 hrs/week).'
            cgpa_color = '#fff8e1'; cgpa_border = '#f9a825'
        else:
            cgpa_msg = 'Low CGPA. Prioritize academics NOW. Max 3 hrs/week freelancing until CGPA improves.'
            cgpa_color = '#fce4ec'; cgpa_border = '#e53935'

        st.markdown(
            f'<div style="background:{cgpa_color}; padding:16px 20px; border-radius:12px; border-left:5px solid {cgpa_border}; margin:10px 0; color:#1b3a1b;">'
            f'<b>CGPA Analysis ({gpa}/4.0 — Semester {semester}):</b><br>{cgpa_msg}</div>',
            unsafe_allow_html=True
        )

        freelance_data = {
            'Python':              {'platforms': 'Upwork, Fiverr',       'services': 'Scripts, automation, web scraping, bots',       'rate': '$10-$50/hr',     'first_gig': '3-5 months',  'career': ['ML Engineer','Software Engineer','Data Scientist']},
            'Web Development':     {'platforms': 'Fiverr, Upwork',       'services': 'Websites, landing pages, React apps, portfolios','rate': '$15-$60/hr',     'first_gig': '2-4 months',  'career': ['Software Engineer','Full Stack Developer','UI/UX Engineer']},
            'AI/ML':               {'platforms': 'Upwork, Toptal',       'services': 'ML models, chatbots, data pipelines',            'rate': '$20-$80/hr',     'first_gig': '5-8 months',  'career': ['ML Engineer','Data Scientist','AI Engineer']},
            'Data Science':        {'platforms': 'Upwork, Kaggle',       'services': 'Data analysis, dashboards, visualizations',      'rate': '$15-$60/hr',     'first_gig': '4-6 months',  'career': ['ML Engineer','Data Analyst','Business Analyst']},
            'Cybersecurity':       {'platforms': 'HackerOne, Upwork',    'services': 'Bug bounty, security audits, pen testing',       'rate': '$20-$100/hr',    'first_gig': '6-12 months', 'career': ['Cybersecurity Expert','Security Analyst','Pen Tester']},
            'Cloud Computing':     {'platforms': 'Upwork, LinkedIn',     'services': 'AWS/GCP setup, cloud migration, DevOps tasks',   'rate': '$25-$80/hr',     'first_gig': '4-8 months',  'career': ['Cloud Architect','DevOps Engineer','Software Engineer']},
            'DevOps':              {'platforms': 'Upwork, Toptal',       'services': 'CI/CD, Docker, Kubernetes, Linux admin',         'rate': '$25-$90/hr',     'first_gig': '6-10 months', 'career': ['DevOps Engineer','Cloud Engineer','Software Engineer']},
            'Java':                {'platforms': 'Upwork, Freelancer',   'services': 'Java apps, Spring Boot APIs, Android',           'rate': '$15-$55/hr',     'first_gig': '3-6 months',  'career': ['Software Engineer','Android Developer','Backend Developer']},
            'Network Security':    {'platforms': 'Upwork, LinkedIn',     'services': 'Network audits, firewall config, VPN setup',     'rate': '$20-$70/hr',     'first_gig': '4-8 months',  'career': ['Cybersecurity Expert','Network Engineer','Security Analyst']},
            'Ethical Hacking':     {'platforms': 'HackerOne, Bugcrowd',  'services': 'Bug bounty, web pentesting, CTF',                'rate': '$20-$100/hr',    'first_gig': '6-12 months', 'career': ['Cybersecurity Expert','Pen Tester','Security Consultant']},
            'Penetration Testing': {'platforms': 'HackerOne, Synack',    'services': 'Web/network/mobile pentesting + reports',        'rate': '$25-$100/hr',    'first_gig': '8-12 months', 'career': ['Pen Tester','Cybersecurity Expert','Red Team Analyst']},
            'Bug Bounty':          {'platforms': 'HackerOne, Bugcrowd',  'services': 'Finding vulnerabilities — earn per bug found',   'rate': '$100-$10,000/bug','first_gig': 'Start on HTB/TryHackMe now','career': ['Security Researcher','Cybersecurity Expert','Pen Tester']},
            'AutoCAD':             {'platforms': 'Fiverr, Upwork',       'services': '2D/3D drawings, architectural plans',            'rate': '$10-$40/hr',     'first_gig': '2-4 months',  'career': ['Civil Engineer','Draftsman','Project Manager']},
            'Marketing':           {'platforms': 'Fiverr, Upwork',       'services': 'Social media, SEO, ad campaigns, email',         'rate': '$8-$30/hr',      'first_gig': '1-3 months',  'career': ['Marketing Executive','Digital Marketer','Business Analyst']},
            'Financial Analysis':  {'platforms': 'Upwork',               'services': 'Business plans, financial models, Excel',        'rate': '$15-$50/hr',     'first_gig': '3-6 months',  'career': ['Financial Analyst','Business Analyst','Accountant']},
            'Accounting':          {'platforms': 'Upwork, Fiverr',       'services': 'Bookkeeping, QuickBooks, tax returns',           'rate': '$10-$35/hr',     'first_gig': '2-4 months',  'career': ['Accountant','Financial Analyst','Tax Consultant']},
            'Graphic Design':      {'platforms': 'Fiverr, 99designs',    'services': 'Logos, banners, social media, branding',         'rate': '$8-$35/hr',      'first_gig': '1-3 months',  'career': ['Graphic Designer','UI/UX Designer','Creative Director']},
            'Content Writing':     {'platforms': 'Fiverr, iWriter',      'services': 'Blog posts, articles, SEO content, copywriting', 'rate': '$5-$25/hr',      'first_gig': '1-2 months',  'career': ['Content Writer','Journalist','Marketing Executive']},
            'Medical Writing':     {'platforms': 'Upwork, Fiverr',       'services': 'Medical articles, patient education content',    'rate': '$10-$35/hr',     'first_gig': '2-4 months',  'career': ['Teacher/Professor','Doctor','Content Writer']},
            'Lab Analysis':        {'platforms': 'Upwork, Fiverr',       'services': 'Lab reports, data entry, research assistance',   'rate': '$8-$25/hr',      'first_gig': '2-3 months',  'career': ['Doctor','Lab Technician','Medical Officer']},
            'Patient Care':        {'platforms': 'Tutor.com, Upwork',    'services': 'Online health tutoring, nursing assistance notes','rate': '$8-$25/hr',      'first_gig': '2-4 months',  'career': ['Doctor','Nurse Practitioner','Lab Technician']},
            'Biology':             {'platforms': 'Tutor.com, Chegg',     'services': 'Online Biology tutoring, assignment help',       'rate': '$10-$30/hr',     'first_gig': '1-2 months',  'career': ['Doctor','Teacher/Professor','Lab Technician']},
            'Anatomy':             {'platforms': 'Tutor.com, Upwork',    'services': 'Online Anatomy tutoring, medical illustration',  'rate': '$10-$30/hr',     'first_gig': '1-2 months',  'career': ['Doctor','Teacher/Professor','Medical Writer']},
            'Pharmacology':        {'platforms': 'Upwork, Fiverr',       'services': 'Pharmacology tutoring, drug info writing',       'rate': '$10-$35/hr',     'first_gig': '2-3 months',  'career': ['Doctor','Pharmacist','Medical Writer']},
            'Surgery Basics':      {'platforms': 'Upwork, Tutor.com',    'services': 'Medical tutoring, surgery notes, case studies',  'rate': '$10-$30/hr',     'first_gig': '2-4 months',  'career': ['Doctor','Surgeon','Medical Officer']},
            'First Aid':           {'platforms': 'Fiverr, Upwork',       'services': 'First aid training content, health writing',     'rate': '$8-$20/hr',      'first_gig': '1-2 months',  'career': ['Doctor','Nurse','Health Educator']},
            'Clinical Skills':     {'platforms': 'Tutor.com, Upwork',    'services': 'Online medical tutoring, case study help',       'rate': '$10-$30/hr',     'first_gig': '1-3 months',  'career': ['Doctor','Medical Officer','Lab Technician']},
            'Freelance Gigs':      {'platforms': 'Fiverr, Upwork',       'services': 'Any skill-based online services',               'rate': '$5-$30/hr',      'first_gig': '2-4 months',  'career': ['Software Engineer','Business Analyst','Content Writer']},
        }

        fl = freelance_data.get(skill, {
            'platforms': 'Fiverr, Upwork', 'services': skill + ' related services',
            'rate': '$10-$40/hr', 'first_gig': '3-6 months',
            'career': ['Software Engineer','Business Analyst','Teacher/Professor']
        })

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            laptop_s = 'You have a laptop' if has_laptop == 'Yes' else 'No laptop — get one! Essential for freelancing'
            net_s = 'Good internet' if internet == 'Good' else ('Average internet — workable' if internet == 'Average' else 'Poor internet — upgrade needed')
            eng_s = 'Fluent English — global high-paying clients!' if english_level == 'Fluent' else ('Intermediate English — you can start now' if english_level == 'Intermediate' else 'Improve English — it will 3x your earning rate')
            st.markdown(
                f'<div class="roadmap-card"><b>Your Setup</b><br><br>'
                f'{"OK" if has_laptop=="Yes" else "!!"} {laptop_s}<br>'
                f'{"OK" if internet!="Poor" else "!!"} {net_s}<br>'
                f'{"OK" if english_level!="Basic" else "--"} {eng_s}<br><br>'
                f'<b>Semester:</b> {semester} | <b>CGPA:</b> {gpa}/4.0<br>'
                f'<b>Free Time:</b> {freelance_time}</div>',
                unsafe_allow_html=True
            )
        with col_r2:
            st.markdown(
                f'<div class="roadmap-card"><b>Freelancing in {skill}</b><br><br>'
                f'<b>Best Platforms:</b><br>{fl["platforms"]}<br><br>'
                f'<b>Services to Offer:</b><br>{fl["services"]}<br><br>'
                f'<b>Expected Rate:</b> {fl["rate"]}<br>'
                f'<b>Time to First Client:</b> {fl["first_gig"]}</div>',
                unsafe_allow_html=True
            )

        if freelance_time == 'Very Less (<5 hrs/week)':
            st.markdown('<div class="warning-box">With less than 5 hrs/week, focus on ONE simple skill. Best options: Content Writing or Graphic Design — quick to learn, easy to start small.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Semester-Based Action Plan")
        if sem_num <= 2:
            steps = [
                ("Build Foundations", "Study your skill 1-2 hrs/day. Use YouTube, freeCodeCamp, Coursera free courses."),
                ("Create Profiles", "Set up Fiverr + Upwork accounts. Study top-rated gigs in your niche."),
                ("Make Samples", "Create 2-3 sample projects to show clients even before your first order."),
            ]
        elif sem_num <= 5:
            steps = [
                ("Build Portfolio", "Create 3-5 real projects with your skill. Host on GitHub, Behance, or your own site."),
                ("Get First Gig", "Post your Fiverr gig at a low rate to collect reviews. Reviews matter more than rate at start!"),
                ("Get Certified", "Earn 1 relevant certification — it adds credibility and justifies higher rates."),
            ]
        else:
            steps = [
                ("Scale Up Rates", "Later semesters — raise your rates. Target $15-50/hr on Upwork now."),
                ("LinkedIn Presence", "Post your work weekly. Connect with 10 new professionals per week. Get referrals."),
                ("Internship + Freelance", "Combine part-time internship with freelancing for maximum experience before graduation."),
            ]
        cols_s = st.columns(3)
        for i, (title, desc) in enumerate(steps):
            with cols_s[i]:
                st.markdown(f'<div class="info-card"><b>{title}</b><br><br><span style="font-size:13px;">{desc}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Earning Projection & Best Platforms")
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            base = 1.3 if gpa >= 3.0 else 0.8
            months_l = ['Month 1','Month 2','Month 3','Month 6','Month 12']
            earnings_l = [0, int(50*base), int(150*base), int(400*base), int(900*base)]
            fig_earn = go.Figure(go.Scatter(
                x=months_l, y=earnings_l, mode='lines+markers+text',
                text=[f'${v}' for v in earnings_l], textposition='top center',
                textfont=dict(color='#4a3728', size=12),
                line=dict(color='#43a047', width=3),
                marker=dict(size=10, color='#43a047'),
                fill='tozeroy', fillcolor='rgba(67,160,71,0.1)'
            ))
            fig_earn.update_layout(
                title=dict(text='Expected Monthly Earnings (USD)', font=dict(color='#4a3728', size=13)),
                plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                font=dict(color='#4a3728', size=12), height=320,
                xaxis=dict(gridcolor='#e0d5c5', tickfont=dict(color='#4a3728')),
                yaxis=dict(title='USD/month', gridcolor='#e0d5c5', tickfont=dict(color='#4a3728'))
            )
            st.plotly_chart(fig_earn, use_container_width=True)
        with col_ch2:
            platform_dist = {
                'Python':[30,40,10,20],'Web Development':[35,35,15,15],'AI/ML':[10,45,5,40],
                'Data Science':[20,40,10,30],'Cybersecurity':[30,30,0,40],'Cloud Computing':[10,40,10,40],
                'DevOps':[10,40,10,40],'Java':[25,35,20,20],'Ethical Hacking':[5,10,5,80],
                'Bug Bounty':[0,0,10,90],'Network Security':[10,35,5,50],'AutoCAD':[40,35,15,10],
                'Graphic Design':[45,25,20,10],'Content Writing':[40,30,20,10],
                'Marketing':[30,30,20,20],'Accounting':[20,40,10,30],'Medical Writing':[30,40,10,20],
            }
            pvals = platform_dist.get(skill, [25,35,15,25])
            fig_plat = go.Figure(go.Pie(
                labels=['Fiverr','Upwork','Freelancer','Direct/Bug Bounty'],
                values=pvals, hole=0.5,
                marker_colors=['#43a047','#2196F3','#f9a825','#8b6f47'],
                textinfo='label+percent', textfont=dict(size=11, color='#4a3728')
            ))
            fig_plat.update_layout(
                title=dict(text='Best Platforms for ' + skill, font=dict(color='#4a3728', size=13)),
                plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                font=dict(color='#4a3728', size=12), height=320
            )
            st.plotly_chart(fig_plat, use_container_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Career Prediction After Graduation")
        pred_careers = fl.get('career', ['Software Engineer','Business Analyst','Teacher/Professor'])
        card_cls_u = ['career-card-gold','career-card-silver','career-card-bronze']
        medals_u = ['1st Choice','2nd Choice','3rd Choice']
        probs_u = [0.65, 0.25, 0.10]
        for i, car in enumerate(pred_careers[:3]):
            ci = career_info.get(car, {'emoji':'💼','desc':'Professional career in your field','salary':'PKR 80,000+/month'})
            pct = int(probs_u[i]*100)
            st.markdown(
                f'<div class="{card_cls_u[i]}">'
                f'<b style="font-size:17px;">{ci.get("emoji","💼")} {car}</b>'
                f'<span style="float:right; font-size:20px; font-weight:bold; color:#8b6f47;">{pct}%</span><br>'
                f'<span style="font-size:13px; color:#7a6652;">{ci.get("desc","")}</span><br>'
                f'<span style="font-size:13px;">Salary: {ci.get("salary","")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Download Your Freelancing Report")
        report_ug = (
            "CAREER PATH PREDICTOR — UNDERGRADUATE REPORT\n"
            "=============================================\n\n"
            "STUDENT PROFILE\n"
            "---------------\n"
            f"Group/Field      : {group}\n"
            f"Semester         : {semester}\n"
            f"CGPA             : {gpa}/4.0\n"
            f"Primary Skill    : {skill}\n"
            f"Certification    : {cert}\n"
            f"Has Laptop       : {has_laptop}\n"
            f"Internet         : {internet}\n"
            f"English Level    : {english_level}\n"
            f"Free Time/Week   : {freelance_time}\n\n"
            "FREELANCING ROADMAP\n"
            "-------------------\n"
            f"Best Platforms   : {fl['platforms']}\n"
            f"Services         : {fl['services']}\n"
            f"Expected Rate    : {fl['rate']}\n"
            f"Time to 1st Gig  : {fl['first_gig']}\n\n"
            "CAREER AFTER GRADUATION\n"
            "-----------------------\n"
        )
        for i, c in enumerate(pred_careers[:3]):
            report_ug += f"{i+1}. {c}\n"
        report_ug += "\nGenerated by Career Path Predictor AI\n"

        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(label="Download My Freelancing Report", data=report_ug,
                               file_name="freelancing_report.txt", mime="text/plain")


    # ===== MATRIC =====
    elif education == 'Matric':
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📋 Your Personalized Career Roadmap")
        suggestion = edu_suggestions.get(group, {})

        st.markdown(f'<div class="warning-box">⚠️ <b>Currently at Matric Level</b><br>Direct AI career prediction requires more education data. Here is your complete roadmap!</div>', unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f'''<div class="roadmap-card">
            🎓 <b>Next Step — Continue Education</b><br><br>
            📌 {suggestion.get("degree", "Continue education")}<br><br>
            🚀 <b>Future Career Options:</b><br>
            {"<br>".join(["• " + c for c in suggestion.get("careers", [])])}
            </div>''', unsafe_allow_html=True)

        with col_r2:
            st.markdown(f'''<div class="roadmap-card">
            💼 <b>Jobs Available Right Now</b><br><br>
            • Primary School Teacher<br>
            • Data Entry Operator<br>
            • Basic Office Work<br>
            • Shop/Business Assistant<br><br>
            <b>Avg Salary:</b> PKR 20,000 - 40,000/month
            </div>''', unsafe_allow_html=True)

        if has_laptop == "Yes" and internet in ["Good", "Average"]:
            st.markdown(f'''<div class="suggestion-box">
            🌟 <b>Freelancing Opportunity Available!</b><br>
            You have a laptop + internet — this is your gateway to earning online!<br><br>
            <b>Recommended Platforms:</b> Fiverr, Upwork, Freelancer.com<br>
            <b>Start Learning:</b> {", ".join(suggestion.get("freelance", ["Basic Computer Skills"]))}<br>
            <b>Expected Earning:</b> $100 - $500/month (after 6 months of learning)
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">💻 <b>Key Investment:</b> Getting a laptop + internet will open doors to online freelancing and much better opportunities!</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Why Education Matters — Job Market Reality")

        fig = go.Figure()
        edu_levels = ['Matric', 'Intermediate', "Bachelor's", "Master's"]
        got_jobs = [25, 45, 72, 85]
        no_jobs = [75, 55, 28, 15]
        colors_got = ['#8b6f47', '#c4956a', '#43a047', '#2196F3']

        fig.add_trace(go.Bar(name='Got Job ✅', x=edu_levels, y=got_jobs,
                            marker_color=colors_got, text=[f'{v}%' for v in got_jobs],
                            textposition='outside', textfont=dict(color='#4a3728', size=14)))
        fig.add_trace(go.Bar(name='No Job ❌', x=edu_levels, y=no_jobs,
                            marker_color='#e0d5c5', text=[f'{v}%' for v in no_jobs],
                            textposition='outside', textfont=dict(color='#7a6652', size=12)))

        fig.update_layout(
            barmode='group',
            plot_bgcolor='#fffdf7',
            paper_bgcolor='#fffdf7',
            font=dict(color='#4a3728', size=13),
            title=dict(text='Employment Rate by Education Level in Pakistan', font=dict(size=16, color='#4a3728')),
            legend=dict(bgcolor='#fffdf7', font=dict(color='#4a3728', size=12)),
            height=400,
            xaxis=dict(gridcolor='#e0d5c5', tickfont=dict(color='#4a3728', size=12)),
            yaxis=dict(gridcolor='#e0d5c5', title='Percentage (%)', tickfont=dict(color='#4a3728', size=12))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="warning-box">📌 <b>Key Insight:</b> Only 25% of Matric graduates find stable employment. Each education level dramatically improves your chances!</div>', unsafe_allow_html=True)

        # Download for Matric
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📄 Download Your Career Report")
        report_matric = f"""
╔══════════════════════════════════════════════════════╗
           🎯 CAREER PATH PREDICTOR — REPORT
╚══════════════════════════════════════════════════════╝

📋 STUDENT PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Education Level  : {education}
Group/Field      : {group}
Percentage       : {percentage}%
Primary Skill    : {skill}
Certification    : {cert}
Has Laptop       : {has_laptop}
Internet Access  : {internet}
City Type        : {city_type}

📚 RECOMMENDED NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next Education   : {suggestion.get("degree", "Continue education")}
Future Careers   : {", ".join(suggestion.get("careers", []))}

💼 JOBS AVAILABLE NOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Primary School Teacher
• Data Entry Operator
• Basic Office Work
• Shop/Business Assistant
Avg Salary       : PKR 20,000 - 40,000/month

💻 FREELANCING (if you have laptop+internet)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skills to Learn  : {", ".join(suggestion.get("freelance", []))}
Platforms        : Fiverr, Upwork, Freelancer.com
Expected Income  : $100 - $500/month (after 6 months)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Generated by Career Path Predictor AI
            Powered by Random Forest ML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(label="📥 Download My Career Report", data=report_matric,
                               file_name=f"career_report_matric_{group.replace(' ','_')}.txt", mime="text/plain")

    # ===== INTERMEDIATE =====
    elif education == 'Intermediate':
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 📋 Your Personalized Career Roadmap")
        suggestion = edu_suggestions.get(group, {})

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f'''<div class="roadmap-card">
            🎓 <b>Recommended Further Education</b><br><br>
            📌 {suggestion.get("degree", "Continue education")}<br><br>
            🚀 <b>Careers After Bachelor\'s:</b><br>
            {"<br>".join(["• " + c for c in suggestion.get("careers", [])])}
            </div>''', unsafe_allow_html=True)

        with col_r2:
            st.markdown(f'''<div class="roadmap-card">
            💼 <b>Current Job Options</b><br><br>
            {"✅ <b>Freelancing in " + skill + "</b> (with your skill)" if cert != "None" else "• Primary School Teacher"}<br>
            • Data Entry Operator<br>
            • Basic Office Assistant<br>
            • Sales/Customer Service<br><br>
            <b>Avg Salary:</b> PKR 25,000 - 60,000/month
            </div>''', unsafe_allow_html=True)

        if cert != 'None' and has_laptop == 'Yes' and internet in ['Good', 'Average']:
            st.markdown(f'''<div class="suggestion-box">
            🌟 <b>You Can Freelance RIGHT NOW!</b><br>
            Certification ✅ + Laptop ✅ + Internet ✅ — Perfect combo!<br><br>
            <b>Your Skill:</b> {skill} | <b>Your Cert:</b> {cert}<br>
            <b>Platforms:</b> Fiverr, Upwork, Freelancer.com<br>
            <b>Expected Earning:</b> $200 - $800/month
            </div>''', unsafe_allow_html=True)
        elif cert == 'None':
            st.markdown(f'''<div class="warning-box">
            💡 <b>One Step Away from Freelancing!</b><br>
            You have <b>{skill}</b> skill but need a certification.<br>
            <b>Get certified in:</b> {", ".join(cert_map.get(group, ["a relevant cert"])[:3])}<br>
            This will unlock freelancing opportunities!
            </div>''', unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Data Insights")
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            edu_levels = ['Matric', 'Intermediate', "Bachelor's", "Master's"]
            got_jobs = [25, 45, 72, 85]
            fig1 = go.Figure(go.Bar(
                x=edu_levels, y=got_jobs,
                marker_color=['#d4a574', '#c4956a', '#8b6f47', '#4a3728'],
                text=[f'{v}%' for v in got_jobs],
                textposition='outside',
                textfont=dict(color='#4a3728', size=13)
            ))
            fig1.update_layout(
                title=dict(text='Employment Rate by Education', font=dict(color='#4a3728', size=14)),
                plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                font=dict(color='#4a3728', size=12), height=350,
                yaxis=dict(title='% Got Job', gridcolor='#e0d5c5', tickfont=dict(color='#4a3728')),
                xaxis=dict(gridcolor='#e0d5c5', tickfont=dict(color='#4a3728'))
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_g2:
            field_careers = {
                'ICS': {'Software Engineer': 45, 'ML Engineer': 25, 'Cybersecurity': 20, 'Other': 10},
                'Pre-Engineering': {'Engineer': 50, 'Project Manager': 25, 'Designer': 15, 'Other': 10},
                'Pre-Medical': {'Doctor': 55, 'Lab Tech': 20, 'Pharmacist': 15, 'Other': 10},
                'Commerce (Inter)': {'Accountant': 40, 'Business Analyst': 30, 'Marketing': 20, 'Other': 10},
                'Arts (Inter)': {'Content Writer': 35, 'Graphic Designer': 35, 'Journalist': 20, 'Other': 10},
            }
            field_data = field_careers.get(group, {'Teacher': 40, 'Data Entry': 35, 'Other': 25})
            fig2 = go.Figure(go.Pie(
                labels=list(field_data.keys()),
                values=list(field_data.values()),
                hole=0.5,
                marker_colors=['#8b6f47', '#c4956a', '#d4a574', '#e8c9a0'],
                textinfo='label+percent',
                textfont=dict(size=12, color='#4a3728')
            ))
            fig2.update_layout(
                title=dict(text=f'Career Paths — {group}', font=dict(color='#4a3728', size=14)),
                plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                font=dict(color='#4a3728', size=12), height=350
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Download for Intermediate
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📄 Download Your Career Report")
        report_inter = f"""
╔══════════════════════════════════════════════════════╗
           🎯 CAREER PATH PREDICTOR — REPORT
╚══════════════════════════════════════════════════════╝

📋 STUDENT PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Education Level  : {education}
Group/Field      : {group}
Percentage       : {percentage}%
Primary Skill    : {skill}
Certification    : {cert}
Has Laptop       : {has_laptop}
Internet Access  : {internet}
City Type        : {city_type}

📚 RECOMMENDED NEXT STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next Education   : {suggestion.get("degree", "Continue education")}
Future Careers   : {", ".join(suggestion.get("careers", []))}

💼 CURRENT JOB OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Data Entry Operator
• Basic Office Assistant
• Sales/Customer Service
• Freelancing (if certified)
Avg Salary       : PKR 25,000 - 60,000/month

💻 FREELANCING PATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your Skill       : {skill}
Your Cert        : {cert}
Skills to Learn  : {", ".join(suggestion.get("freelance", []))}
Platforms        : Fiverr, Upwork, Freelancer.com
Expected Income  : $200 - $800/month (with certification)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      Generated by Career Path Predictor AI
            Powered by Random Forest ML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(label="📥 Download My Career Report", data=report_inter,
                               file_name=f"career_report_inter_{group.replace(' ','_')}.txt", mime="text/plain")

    # ===== BACHELOR'S (COMPLETED) / MASTER'S =====
    else:
        try:
            edu_enc = le_edu.transform([education])[0] if education in le_edu.classes_ else 0
            group_enc = le_group.transform([group])[0] if group in le_group.classes_ else 0
            skill_enc = le_skill.transform([skill])[0] if skill in le_skill.classes_ else 0
            interest_enc = le_interest.transform([interest_area])[0] if interest_area in le_interest.classes_ else 0
            work_enc = le_work.transform([work_pref])[0] if work_pref in le_work.classes_ else 0

            input_data = pd.DataFrame(
                [[edu_enc, group_enc, skill_enc, interest_enc, work_enc, cgpa]],
                columns=['edu_enc', 'group_enc', 'skill_enc', 'interest_enc', 'work_enc', 'cgpa']
            )

            proba = model.predict_proba(input_data)[0]
            top3_idx = np.argsort(proba)[::-1][:3]
            top3_careers = list(le_career.inverse_transform(top3_idx))
            top3_probs = list(proba[top3_idx])

            # Rule-based override — Medical FIXED
            if group in ['CS/IT', 'ICS', 'MS/MPhil CS'] and interest_area == 'Technology':
                if skill in ['AI/ML', 'Data Science', 'Advanced AI', 'Deep Learning', 'NLP', 'Big Data']:
                    top3_careers = ['ML Engineer', 'Software Engineer', 'Cybersecurity Expert']
                elif skill in ['Cybersecurity', 'Cloud Computing', 'Networking']:
                    top3_careers = ['Cybersecurity Expert', 'Software Engineer', 'ML Engineer']
                else:
                    top3_careers = ['Software Engineer', 'ML Engineer', 'Cybersecurity Expert']
                top3_probs = [0.70, 0.20, 0.10]
            elif group in ['Cybersecurity (BS)', 'MS Cybersecurity']:
                if skill in ['Advanced Penetration Testing','Digital Forensics','Threat Intelligence','Security Architecture','Zero Trust','Malware Reverse Engineering']:
                    top3_careers = ['Cybersecurity Expert', 'ML Engineer', 'Software Engineer']
                    top3_probs = [0.80, 0.12, 0.08]
                else:
                    top3_careers = ['Cybersecurity Expert', 'Software Engineer', 'ML Engineer']
                    top3_probs = [0.80, 0.12, 0.08]
            elif group in ['Pre-Medical', 'Medical/Health', 'MS Medical']:
                if interest_area == 'Healthcare':
                    top3_careers = ['Doctor', 'Teacher/Professor', 'Business Analyst']
                    top3_probs = [0.80, 0.12, 0.08]
                elif interest_area == 'Education & Research':
                    top3_careers = ['Teacher/Professor', 'Doctor', 'Content Writer']
                    top3_probs = [0.70, 0.20, 0.10]
                else:
                    top3_careers = ['Doctor', 'Teacher/Professor', 'Content Writer']
                    top3_probs = [0.75, 0.15, 0.10]
            elif group in ['Business/Commerce', 'MBA', 'Commerce (Inter)'] and interest_area == 'Business':
                if skill in ['Accounting', 'Tally ERP', 'Finance', 'Supply Chain']:
                    top3_careers = ['Accountant', 'Financial Analyst', 'Business Analyst']
                elif skill in ['Marketing', 'HR Management', 'E-commerce']:
                    top3_careers = ['Marketing Executive', 'Business Analyst', 'Financial Analyst']
                else:
                    top3_careers = ['Business Analyst', 'Financial Analyst', 'Marketing Executive']
                top3_probs = [0.65, 0.25, 0.10]
            elif group in ['Arts/Humanities', 'Arts (Inter)', 'MA/MSc Arts'] and interest_area == 'Arts & Media':
                if skill in ['Graphic Design', 'Fine Arts', 'Photography', 'Media Production']:
                    top3_careers = ['Graphic Designer', 'Content Writer', 'Marketing Executive']
                else:
                    top3_careers = ['Content Writer', 'Marketing Executive', 'Graphic Designer']
                top3_probs = [0.65, 0.25, 0.10]
            elif group == 'Law':
                top3_careers = ['Lawyer', 'Teacher/Professor', 'Business Analyst']
                top3_probs = [0.75, 0.15, 0.10]
            elif group in ['Education'] and interest_area == 'Education & Research':
                top3_careers = ['Teacher/Professor', 'Content Writer', 'Business Analyst']
                top3_probs = [0.75, 0.15, 0.10]
            elif group == 'Engineering' and interest_area == 'Technology':
                top3_careers = ['Software Engineer', 'Business Analyst', 'Teacher/Professor']
                top3_probs = [0.65, 0.25, 0.10]

            best = top3_careers[0]
            info = career_info.get(best, {})

            # RESULTS HEADER
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("## 🏆 Your AI Career Analysis")

            st.markdown(f'<div class="result-box">{info.get("emoji","🎯")} Best Career Match: {best}</div>', unsafe_allow_html=True)

            # Top 3 cards
            card_styles = ['career-card-gold', 'career-card-silver', 'career-card-bronze']
            medals = ["🥇", "🥈", "🥉"]
            for i in range(3):
                ci = career_info.get(top3_careers[i], {})
                st.markdown(f'''<div class="{card_styles[i]}">
                {medals[i]} <b style="font-size:18px;">{ci.get("emoji","💼")} {top3_careers[i]}</b>
                <span style="float:right; font-size:22px; font-weight:bold; color:#8b6f47;">{top3_probs[i]*100:.0f}%</span><br>
                <span style="font-size:13px; color:#7a6652;">{ci.get("desc","")}</span><br>
                <span style="font-size:13px;">💰 {ci.get("salary","")}</span>
                </div>''', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Gauge chart
            col_gauge1, col_gauge2, col_gauge3 = st.columns(3)
            for i, (col, career, prob) in enumerate(zip([col_gauge1, col_gauge2, col_gauge3], top3_careers, top3_probs)):
                with col:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={'text': f"{medals[i]} {career}", 'font': {'size': 13, 'color': '#4a3728'}},
                        number={'suffix': '%', 'font': {'color': '#8b6f47', 'size': 24}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#4a3728', 'tickfont': {'color': '#4a3728'}},
                            'bar': {'color': ['#f9a825', '#9e9e9e', '#c4956a'][i]},
                            'bgcolor': '#f5f0e8',
                            'bordercolor': '#e0d5c5',
                            'steps': [
                                {'range': [0, 30], 'color': '#f5f0e8'},
                                {'range': [30, 70], 'color': '#e8d5c0'},
                                {'range': [70, 100], 'color': '#d4a574'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        height=220, paper_bgcolor='#fffdf7',
                        font=dict(color='#4a3728', size=12),
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Personalized Tips
            tips_shown = False
            st.markdown("### 💡 Personalized Tips for You:")

            if go_abroad == "Yes" and english_level == "Fluent":
                st.markdown('<div class="suggestion-box">✈️ <b>Abroad Opportunity:</b> Fluent English + willing to relocate — Gulf, UK or Canada are great options!</div>', unsafe_allow_html=True)
                tips_shown = True
            if city_type == "Village/Rural Area" and work_pref in ["Remote", "Any"]:
                st.markdown('<div class="suggestion-box">🌐 <b>Remote Work:</b> You can earn globally from anywhere — focus on freelancing!</div>', unsafe_allow_html=True)
                tips_shown = True
            if job_priority == "High Salary" and best in ['ML Engineer', 'Software Engineer', 'Cybersecurity Expert']:
                st.markdown('<div class="suggestion-box">💰 <b>Salary Tip:</b> Add international certifications to maximize earning potential — can reach PKR 500,000+/month!</div>', unsafe_allow_html=True)
                tips_shown = True
            if gpa_trend == "Declining":
                st.markdown('<div class="warning-box">⚠️ <b>Academic Alert:</b> Declining grades can affect job prospects — seek help and improve ASAP!</div>', unsafe_allow_html=True)
                tips_shown = True
            if gpa_trend == "Improving":
                st.markdown('<div class="suggestion-box">📈 <b>Keep it up!</b> Improving grades make a strong impression on employers and scholarship committees!</div>', unsafe_allow_html=True)
                tips_shown = True
            if extracurricular == "Debates/Public Speaking" and best in ['Marketing Executive', 'Lawyer', 'Teacher/Professor']:
                st.markdown('<div class="suggestion-box">🎤 <b>Perfect Match:</b> Your public speaking skills directly align with your career — highlight this in interviews!</div>', unsafe_allow_html=True)
                tips_shown = True
            if relocate == "Yes" and "Small City" in city_type:
                st.markdown('<div class="suggestion-box">🏙️ <b>Move for Opportunity:</b> Karachi, Lahore or Islamabad will dramatically expand your job options!</div>', unsafe_allow_html=True)
                tips_shown = True
            if cert != 'None':
                st.markdown(f'<div class="suggestion-box">📜 <b>Certification Edge:</b> Your <b>{cert}</b> sets you apart from other candidates!</div>', unsafe_allow_html=True)
                tips_shown = True
            if not tips_shown:
                st.markdown('<div class="suggestion-box">✅ <b>Good Profile!</b> Focus on building practical experience through internships and projects!</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Why recommended
            st.markdown("### 🔍 Prediction Basis")
            cols_basis = st.columns(3)
            basis_items = [
                (f"🎓 {education}", group),
                (f"📊 GPA", f"{gpa:.1f}/4.0"),
                (f"💡 Skill", skill),
                (f"🌟 Interest", interest_area),
                (f"💼 Work Pref", work_pref),
                (f"🌍 City", city_type.split('(')[0].strip())
            ]
            for i, (label, value) in enumerate(basis_items):
                with cols_basis[i % 3]:
                    st.markdown(f'<div class="basis-card"><b>{label}</b><br>{value}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Career Details
            st.markdown("### 📋 Best Match — Full Career Profile")
            if info:
                col_i1, col_i2, col_i3 = st.columns(3)
                with col_i1:
                    st.markdown(f'<div class="info-card">📝 <b>Job Description</b><br><br>{info["desc"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card">💰 <b>Salary in Pakistan</b><br><br>{info["salary"]}</div>', unsafe_allow_html=True)
                with col_i2:
                    st.markdown(f'<div class="info-card">🎓 <b>Required Degree</b><br><br>{info["degree"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-card">🔮 <b>Future Scope</b><br><br>{info["scope"]}</div>', unsafe_allow_html=True)
                with col_i3:
                    st.markdown(f'<div class="info-card">📍 <b>Job Areas</b><br><br>{info["areas"]}</div>', unsafe_allow_html=True)
                    growth = info.get('growth', 70)
                    fig_growth = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=growth,
                        title={'text': "📈 Growth Index", 'font': {'size': 12, 'color': '#4a3728'}},
                        number={'suffix': '/100', 'font': {'color': '#8b6f47', 'size': 20}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickfont': {'color': '#4a3728'}},
                            'bar': {'color': '#8b6f47'},
                            'steps': [
                                {'range': [0, 40], 'color': '#f5f0e8'},
                                {'range': [40, 70], 'color': '#e8c9a0'},
                                {'range': [70, 100], 'color': '#d4a574'}
                            ],
                        }
                    ))
                    fig_growth.update_layout(
                        height=180, paper_bgcolor='#fffdf7',
                        font=dict(color='#4a3728', size=12),
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_growth, use_container_width=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Charts
            st.markdown("### 📊 Data Analytics")
            col_g1, col_g2, col_g3 = st.columns(3)

            with col_g1:
                similar = df[df['group'] == group]['career'].value_counts()
                fig1 = go.Figure(go.Pie(
                    labels=similar.index.tolist(),
                    values=similar.values.tolist(),
                    hole=0.5,
                    marker_colors=['#2196F3','#1565C0','#42A5F5','#E53935','#43A047','#81C784','#f9a825','#c4956a','#8b6f47','#d4a574','#66BB6A','#1976D2'],
                    textinfo='label+percent',
                    textfont=dict(size=10, color='#4a3728')
                ))
                fig1.update_layout(
                    title=dict(text=f'Students from {group}', font=dict(color='#4a3728', size=13)),
                    plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                    font=dict(color='#4a3728', size=12), height=350,
                    showlegend=False
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col_g2:
                fig2 = go.Figure(go.Bar(
                    x=[p*100 for p in top3_probs],
                    y=top3_careers,
                    orientation='h',
                    marker_color=['#f9a825', '#9e9e9e', '#c4956a'],
                    text=[f'{p*100:.0f}%' for p in top3_probs],
                    textposition='outside',
                    textfont=dict(color='#4a3728', size=14)
                ))
                fig2.update_layout(
                    title=dict(text='Your Match Scores', font=dict(color='#4a3728', size=13)),
                    plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                    font=dict(color='#4a3728', size=12), height=350,
                    xaxis=dict(range=[0, 100], title='Match %', gridcolor='#e0d5c5', tickfont=dict(color='#4a3728')),
                    yaxis=dict(gridcolor='#e0d5c5', tickfont=dict(color='#4a3728'))
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col_g3:
                edu_levels = ['Matric', 'Intermediate', "Bachelor's", "Master's"]
                got_jobs = [25, 45, 72, 85]
                fig3 = go.Figure(go.Scatter(
                    x=edu_levels, y=got_jobs,
                    mode='lines+markers+text',
                    text=[f'{v}%' for v in got_jobs],
                    textposition='top center',
                    textfont=dict(color='#4a3728', size=12),
                    line=dict(color='#8b6f47', width=3),
                    marker=dict(size=12, color='#c4956a', line=dict(color='#4a3728', width=2)),
                    fill='tozeroy',
                    fillcolor='rgba(139,111,71,0.1)'
                ))
                fig3.update_layout(
                    title=dict(text='Employment Rate Trend', font=dict(color='#4a3728', size=13)),
                    plot_bgcolor='#fffdf7', paper_bgcolor='#fffdf7',
                    font=dict(color='#4a3728', size=12), height=350,
                    xaxis=dict(gridcolor='#e0d5c5', tickfont=dict(color='#4a3728')),
                    yaxis=dict(title='% Employed', gridcolor='#e0d5c5', range=[0, 100], tickfont=dict(color='#4a3728'))
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Download Report
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📄 Download Your Career Report")

            report = f"""
╔══════════════════════════════════════════════════════╗
           🎯 CAREER PATH PREDICTOR — REPORT
╚══════════════════════════════════════════════════════╝

📋 STUDENT PROFILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Education Level  : {education}
Group/Field      : {group}
GPA              : {gpa:.1f}/4.0
Primary Skill    : {skill}
Certification    : {cert}
Interest Area    : {interest_area}
Work Preference  : {work_pref}
City Type        : {city_type}
English Level    : {english_level}

🏆 AI CAREER RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🥇 Best Match    : {top3_careers[0]} ({top3_probs[0]*100:.0f}%)
🥈 Second Match  : {top3_careers[1]} ({top3_probs[1]*100:.0f}%)
🥉 Third Match   : {top3_careers[2]} ({top3_probs[2]*100:.0f}%)

📊 BEST CAREER DETAILS — {best}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Description      : {info.get('desc','')}
Salary Range     : {info.get('salary','')}
Required Degree  : {info.get('degree','')}
Future Scope     : {info.get('scope','')}
Job Areas        : {info.get('areas','')}
Growth Index     : {info.get('growth',0)}/100

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Generated by Career Path Predictor AI
              Powered by Random Forest ML
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            """

            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="📥 Download My Career Report",
                    data=report,
                    file_name=f"career_report_{best.replace(' ','_')}.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"Error: {e}")