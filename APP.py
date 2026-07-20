"""
╔══════════════════════════════════════════════════════════════╗
║   CAREER RECOMMENDATION SYSTEM — v3.0 (FIXED + UPGRADED)   ║
║   For Students After 10th, 12th, and Engineering            ║
║                                                              ║
║   FIXES in v3.0:                                            ║
║   • "Number of results" label corrected to "Top Results"    ║
║   • Slider range changed from 1-10 → 1-100                  ║
║   • Added domain-specific charts: Engineering, Medical,     ║
║     Science Research, Arts & Design                         ║
║   • All streams fully covered in interactive charts         ║
╚══════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
# SECTION 1 │ IMPORTS
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────
# SECTION 2 │ DATASET
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), "career_dataset_v2.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        st.error("❌ career_dataset_v2.csv not found! Keep it in the same folder as app.py.")
        st.stop()
    df["skills"]       = df["skills"].str.lower().str.strip()
    df["description"]  = df["description"].fillna("")
    df["exam_required"]= df["exam_required"].fillna("None")
    df["top_colleges"] = df["top_colleges"].fillna("Various institutions")
    df["category"]     = df["category"].fillna("General")
    df["after_class"]  = df["after_class"].fillna("12th")
    df["profile"]      = df["skills"] + " " + df["description"].str.lower() + " " + df["category"].str.lower()
    return df

@st.cache_resource
def build_tfidf_model(_df):
    vectorizer   = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(_df["profile"])
    return vectorizer, tfidf_matrix

def get_recommendations(user_text, df, vectorizer, tfidf_matrix,
                        stream_filter=None, category_filter=None, top_n=5):
    if not user_text.strip():
        return pd.DataFrame()
    user_vector = vectorizer.transform([user_text.lower()])
    scores      = cosine_similarity(user_vector, tfidf_matrix).flatten()
    result_df   = df.copy()
    result_df["similarity_score"] = scores
    if stream_filter and stream_filter != "All":
        result_df = result_df[result_df["stream"].str.lower() == stream_filter.lower()]
    if category_filter and category_filter != "All Categories":
        result_df = result_df[result_df["category"] == category_filter]
    return result_df.sort_values("similarity_score", ascending=False).head(top_n).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# SECTION 3 │ 10TH GRADE STREAM LOGIC
# ─────────────────────────────────────────────────────────────
def recommend_stream_for_10th(math, science, english, social, interests, skills):
    scores = {"Science": 0.0, "Commerce": 0.0, "Arts": 0.0}
    if math    >= 85: scores["Science"]  += 3
    elif math  >= 70: scores["Science"]  += 2
    elif math  >= 55: scores["Science"]  += 1
    if science >= 85: scores["Science"]  += 3
    elif science >= 70: scores["Science"] += 2
    elif science >= 55: scores["Science"] += 1
    if math    >= 60: scores["Commerce"] += 2
    elif math  >= 45: scores["Commerce"] += 1
    if english >= 65: scores["Commerce"] += 2
    elif english >= 50: scores["Commerce"] += 1
    if english >= 70: scores["Arts"]     += 3
    elif english >= 55: scores["Arts"]   += 2
    elif english >= 40: scores["Arts"]   += 1
    if social  >= 70: scores["Arts"]     += 2
    elif social >= 55: scores["Arts"]    += 1
    scores["Arts"] += 0.5

    stream_profiles = {
        "Science":  "mathematics physics chemistry biology engineering technology medicine research programming computer",
        "Commerce": "business finance accounting economics management marketing banking trade money investment",
        "Arts":     "history literature art music drawing language communication writing law social psychology design"
    }
    user_text = f"{interests} {skills}".lower()
    sv  = TfidfVectorizer(stop_words="english")
    sm  = sv.fit_transform(list(stream_profiles.values()))
    uv  = sv.transform([user_text])
    sim = cosine_similarity(uv, sm).flatten()
    for i, s in enumerate(stream_profiles.keys()):
        scores[s] += sim[i] * 3

    recommended = max(scores, key=scores.get)
    config = {
        "Science":  {"emoji":"🔬","color":"#00b4d8","bg":"#023e8a",
                     "careers":"Engineering (JEE), Medicine (NEET), Data Science, Architecture, Research, ISRO/DRDO",
                     "exams":"JEE Main, JEE Advanced, NEET, BITSAT, KVPY, NDA (PCM)",
                     "reasons":[f"Strong Math ({math}%) and Science ({science}%) marks",
                                "Interests align with technical/analytical fields",
                                "Highest salary potential and global demand"]},
        "Commerce": {"emoji":"📊","color":"#2dc653","bg":"#1a4731",
                     "careers":"CA, MBA, Banking (IBPS), Investment Banking, Marketing, SSC CGL",
                     "exams":"CA Foundation, CUET Commerce, CLAT, Bank PO, SSC CGL",
                     "reasons":[f"Good Math ({math}%) suitable for finance and business",
                                "Interests align with business and economics",
                                "Excellent scope in Banking, Finance, and Management"]},
        "Arts":     {"emoji":"🎨","color":"#e07a5f","bg":"#6b2d1e",
                     "careers":"Law (CLAT), Journalism, Design (NID/NIFT), Civil Services, Psychology, Teaching",
                     "exams":"CLAT, NIFT, NID, IIMC, CUET Arts, UPSC (later)",
                     "reasons":[f"Strong English ({english}%) and social science marks",
                                "Interests align with creative and social fields",
                                "Excellent for Civil Services, Law, Media, and Design"]}
    }
    return {"stream": recommended, "scores": scores,
            "config": config[recommended], "interest_sim": dict(zip(stream_profiles.keys(), sim))}

# ─────────────────────────────────────────────────────────────
# SECTION 4 │ OPPORTUNITIES DATA
# ─────────────────────────────────────────────────────────────
OPPORTUNITIES_10TH = {
    "🔬 After 10th → Science (PCM/PCB)": [
        ("JEE Main / Advanced","Engineering → IITs, NITs, IIITs"),
        ("NEET-UG","Medical → MBBS, BDS, BAMS, BHMS"),
        ("BITSAT","BITS Pilani engineering admission"),
        ("NDA (start prep now)","Join Indian Army, Navy, Air Force as officer"),
        ("KVPY / NTSE","Scholarship for science research careers"),
        ("Polytechnic Diploma","3-year engineering diploma — direct entry to jobs"),
        ("ITI Courses","Vocational: Electrician, Fitter, Welder, AC Mechanic"),
    ],
    "📊 After 10th → Commerce": [
        ("CA Foundation","Start CA journey right after 10th — ICAI exam"),
        ("CS Foundation","Company Secretary foundation course — ICSI"),
        ("CMA Foundation","Cost Management Accountant foundation"),
        ("BBA Entrance (IPMAT)","Integrated BBA+MBA at IIMs — 5 year program"),
        ("Diploma in Banking / Finance","Short skill courses for finance sector jobs"),
    ],
    "🎨 After 10th → Arts / Humanities": [
        ("CLAT (after 12th, start prep)","Law colleges (NLUs) — 5 year LLB"),
        ("NIFT (after 12th)","Fashion design — 19 NIFT campuses across India"),
        ("NID (after 12th)","Product/graphic design — National Institute of Design"),
        ("IIMC (after 12th)","Journalism — Indian Institute of Mass Communication"),
        ("Fine Arts / BFA","Drawing, Painting, Sculpture, Animation"),
    ],
    "🏛️ General After 10th": [
        ("SSC CHSL (after 12th)","Central government jobs — LDC, DEO, PA"),
        ("Indian Army Soldier GD","Soldier posts after 10th in selected categories"),
        ("Railway Group D / NTPC","Railway jobs — start appearing after 12th"),
        ("State Police Constable","State-level police recruitment after 12th"),
        ("Skill India / PMKVY","Free government vocational training programs"),
    ]
}

OPPORTUNITIES_12TH = {
    "🔬 Science → Engineering": [
        ("JEE Main","NITs, IIITs, CFTIs — 31 lakh+ students appear annually"),
        ("JEE Advanced","IITs only — top 2.5 lakh JEE Main qualifiers eligible"),
        ("BITSAT","BITS Pilani, Goa, Hyderabad B.Tech admission"),
        ("VITEEE","VIT University — multiple campuses"),
        ("MHT-CET / KCET / AP EAPCET","State engineering entrances"),
        ("GATE (after B.Tech)","M.Tech at IITs/NITs + PSU jobs (ISRO, ONGC, BHEL)"),
    ],
    "🏥 Science → Medical": [
        ("NEET-UG","MBBS, BDS, BAMS, BHMS, BVSc, B.Pharm — single exam"),
        ("AIIMS MBBS","Now merged with NEET — target AIIMS Delhi, Jodhpur etc."),
        ("JIPMER","Jawaharlal Inst. of Postgraduate Medical Education"),
        ("B.Pharm","4-year pharmacy after PCB/PCM"),
        ("B.Sc Nursing","3-year nursing — AIIMS, CMC Vellore, govt hospitals"),
        ("NEET-PG (after MBBS)","MD/MS specialization in hospitals"),
    ],
    "✈️ Science → Defence & Aviation": [
        ("NDA","Join Army/Navy/Air Force after 12th PCM — UPSC exam"),
        ("Indian Navy SSR / AA","Join Navy as sailor after 12th PCM/PCB"),
        ("Air Force AFCAT","Join Air Force as officer — any graduate"),
        ("Merchant Navy (IMU CET)","Work on international ships after 12th PCM"),
        ("Commercial Pilot (DGCA CPL)","Fly commercial airlines — 12th PCM minimum"),
        ("Coast Guard Navik","Join Indian Coast Guard after 12th"),
    ],
    "📊 Commerce → Finance & Management": [
        ("CA Foundation","ICAI Chartered Accountant — start right after 12th"),
        ("CS Foundation","Company Secretary — ICSI exam"),
        ("BBA / B.Com / BAF","Bachelor's in Business, Commerce, or Accounting"),
        ("IPMAT (IIM Indore/Rohtak)","Integrated MBA program — 5 years"),
        ("CUET","Central University entrance for B.Com, BBA, Economics"),
        ("Bank PO (after graduation)","IBPS PO, SBI PO — start preparation now"),
    ],
    "⚖️ Arts → Law, Media & Design": [
        ("CLAT","Common Law Admission Test — NLUs across India, 5-year LLB"),
        ("AILET","National Law University Delhi entrance"),
        ("NIFT Entrance","Fashion design at 19 NIFT campuses"),
        ("NID Entrance","Product, graphic, animation design at NID"),
        ("IIMC Entrance","Journalism — print, TV, digital media"),
        ("CUET Arts","BA History, Sociology, Pol. Science at central universities"),
        ("BFA / Fine Arts","Bachelor of Fine Arts — drawing, painting, sculpture"),
    ],
    "🏛️ Government Jobs After 12th": [
        ("SSC CHSL (12th pass)","LDC, DEO, PA in central govt — SSC exam"),
        ("Indian Army Soldier GD","After 12th — infantry, technical, clerk posts"),
        ("Railway NTPC (after 12th)","Railway clerk, traffic assistant — RRB exam"),
        ("State Police Constable","State-level police — written + physical test"),
        ("Post Office GDS","Gramin Dak Sevak — India Post, after 10th/12th"),
        ("CRPF / BSF Constable","Paramilitary forces — after 12th"),
    ],
    "🚀 Engineering Branches (After 12th PCM)": [
        ("Computer Science & IT","Software, AI/ML, Cybersecurity — highest demand"),
        ("Electronics & Communication","VLSI, embedded systems, telecom, IoT"),
        ("Mechanical Engineering","Manufacturing, automotive, aerospace, robotics"),
        ("Civil Engineering","Infrastructure, construction, urban planning"),
        ("Electrical Engineering","Power systems, renewable energy, smart grids"),
        ("Chemical Engineering","Petroleum, pharma, food processing industries"),
        ("Aerospace Engineering","ISRO, HAL, defence aviation — very specialized"),
        ("Biomedical Engineering","Medical devices, healthcare tech, prosthetics"),
        ("Artificial Intelligence (New)","ML, deep learning, NLP — emerging branch"),
        ("Data Science Engineering","Analytics, big data, business intelligence"),
        ("Environmental Engineering","Water treatment, pollution control"),
        ("Robotics & Automation","Industrial robots, drones, autonomous vehicles"),
        ("Petroleum Engineering","Oil & gas — ONGC, Reliance, Shell"),
        ("Naval Architecture","Ship design — IIT Madras, Cochin University"),
        ("Agricultural Engineering","Farm machinery, irrigation, food tech"),
        ("Food Technology","Food processing, quality control, FSSAI"),
        ("Textile Engineering","Textile manufacturing and fibre technology"),
        ("Mining Engineering","Mineral extraction — IIT ISM Dhanbad"),
        ("Biotechnology Engineering","Genetic engineering, biopharma, agriculture"),
        ("Information Technology","Web dev, databases, enterprise software"),
    ]
}

GOVT_JOB_CATEGORIES = {
    "🏛️ Civil Services (UPSC)": {
        "desc":"India's most prestigious exams for IAS, IPS, IFS, IRS officers",
        "exams":["UPSC CSE (Prelims → Mains → Interview)","UPSC IFS (Forest Service)","UPSC IES (Engineering Services)","UPSC CDS (Defence)","UPSC CAPF (Paramilitary)"],
        "eligibility":"Graduation in any stream. Age: 21–32 (Gen), up to 37 for SC/ST",
        "salary":"₹56,100 – ₹2,50,000/month + allowances","prep_time":"1–3 years"},
    "🏦 Banking Sector": {
        "desc":"Lakhs of vacancies in nationalized banks, RBI, NABARD every year",
        "exams":["IBPS PO","IBPS Clerk","IBPS SO (Specialist Officer)","SBI PO","SBI Clerk","RBI Grade B","RBI Assistant","NABARD Grade A/B"],
        "eligibility":"Graduation (any stream). Age: 20–28/30 (varies)",
        "salary":"₹25,000 – ₹1,00,000/month","prep_time":"6–18 months"},
    "📋 SSC (Staff Selection Commission)": {
        "desc":"Central government jobs in ministries, CBI, Income Tax, Customs",
        "exams":["SSC CGL (Graduate level)","SSC CHSL (12th pass)","SSC MTS (10th pass)","SSC JE (Junior Engineer)","SSC CPO (Sub-Inspector)"],
        "eligibility":"10th/12th/Graduation depending on post",
        "salary":"₹18,000 – ₹80,000/month","prep_time":"6–12 months"},
    "🚂 Railways (RRB)": {
        "desc":"India's largest employer — lakhs of vacancies annually",
        "exams":["RRB NTPC (Graduate)","RRB Group D (10th pass)","RRB JE (Junior Engineer)","RRB ALP (Assistant Loco Pilot)"],
        "eligibility":"10th/ITI/Diploma/Graduation depending on post",
        "salary":"₹18,000 – ₹75,000/month","prep_time":"6–12 months"},
    "🛡️ Defence & Paramilitary": {
        "desc":"Serve the nation in Army, Navy, Air Force, and paramilitary",
        "exams":["NDA (after 12th PCM)","CDS (after graduation)","AFCAT (Air Force)","SSC GD Constable","CRPF/BSF/CISF/ITBP/SSB"],
        "eligibility":"10th/12th/Graduation + physical fitness tests",
        "salary":"₹21,700 – ₹2,50,000/month","prep_time":"6–24 months"},
    "🔬 PSUs (Public Sector Undertakings)": {
        "desc":"ISRO, DRDO, ONGC, BHEL, NTPC, BARC — prestigious govt companies",
        "exams":["GATE (engineering PSU recruitment)","ISRO Centralized Recruitment","DRDO CEPTAM","ONGC/BHEL/HPCL written tests","BARC"],
        "eligibility":"B.Tech/B.Sc depending on PSU. GATE score is key",
        "salary":"₹40,000 – ₹2,00,000/month","prep_time":"GATE: 6–18 months"},
    "🏫 Teaching & Education": {
        "desc":"Teach in government schools, KV, NVS, and universities",
        "exams":["CTET","State TET (HTET, MAHA TET, UP TET)","UGC-NET (for college teaching)","KVS/NVS Teacher Recruitment","DSSSB"],
        "eligibility":"B.Ed + Graduation for school. M.A/M.Sc + NET for college",
        "salary":"₹28,000 – ₹1,50,000/month","prep_time":"6–18 months"},
}

# ─────────────────────────────────────────────────────────────
# SECTION 5 │ DOMAIN CHART DATA (Engineering, Medical, Arts, Science)
# ─────────────────────────────────────────────────────────────

ENGINEERING_BRANCHES = {
    "Branch": [
        "Computer Science & IT","AI / Machine Learning","Electronics & Communication",
        "Mechanical Engineering","Civil Engineering","Electrical Engineering",
        "Aerospace Engineering","Chemical Engineering","Biomedical Engineering",
        "Robotics & Automation","Data Science Engg","Cybersecurity",
        "Petroleum Engineering","Environmental Engg","Naval Architecture",
        "Mining Engineering","Agricultural Engg","Food Technology",
        "Textile Engineering","Biotechnology Engg"
    ],
    "Avg Salary (LPA)": [12,18,10,8,7,9,14,10,9,13,16,15,18,7,10,9,6,6,5,8],
    "Job Demand (%)": [95,98,85,80,75,82,70,72,68,88,94,92,65,60,55,58,65,62,55,70],
    "Top Exam": [
        "JEE/GATE","JEE/GATE","JEE/GATE","JEE/GATE","JEE/GATE","JEE/GATE",
        "JEE/GATE","JEE/GATE","JEE/GATE","JEE/GATE","JEE/GATE","CEH/JEE",
        "JEE/GATE","JEE/GATE","JEE/GATE","JEE/GATE","JEE/ICAR","JEE/GATE",
        "JEE/GATE","JEE/GATE"
    ],
    "Key Recruiter": [
        "TCS, Google, Amazon","FAANG, Startups","Qualcomm, Samsung","Tata Motors, L&T","L&T, NHAI","NTPC, Power Grid",
        "ISRO, HAL, Boeing","HPCL, Reliance","Apollo, Medtronic","ABB, Bosch","Infosys, Wipro","Govt/Banks/Tech",
        "ONGC, Shell","CPCB, Env. Agencies","Shipping Corps","Coal India, SAIL","State Agri Depts","Nestle, ITC",
        "Raymond, Welspun","Biocon, Dr. Reddy"
    ]
}

MEDICAL_CAREERS = {
    "Career": [
        "Doctor (MBBS)","Surgeon (MS/MD)","Dentist (BDS)","Ayurvedic Doctor (BAMS)",
        "Homeopathic Doctor (BHMS)","Pharmacist (B.Pharm)","Nurse (B.Sc Nursing)",
        "Physiotherapist (BPT)","Radiologist","Veterinary Doctor (BVSc)",
        "Nutritionist / Dietitian","Medical Lab Technician","Occupational Therapist",
        "Speech Therapist","Optometrist","Paramedic / EMT","Public Health Officer",
        "Biomedical Engineer"
    ],
    "Avg Salary (LPA)": [12,30,10,6,5,5,4,6,25,8,5,4,5,5,5,4,8,10],
    "Study Duration (Yrs)": [5.5,8,5,5.5,5.5,4,3,4.5,10,5,3,2,4,4,4,2,5,4],
    "Exam": [
        "NEET-UG","NEET-UG+PG","NEET-UG","NEET-UG","NEET-UG","GPAT",
        "State Nursing","NEET (some)","NEET-UG+PG","NEET-UG","State exam",
        "State DMLT","State exam","State exam","State exam","State exam","UPSC/State","JEE/GATE"
    ],
    "Job Setting": [
        "Hospital/Clinic","Hospital","Dental Clinic","Clinic/Hospital","Clinic","Pharmacy/Hospital",
        "Hospital","Clinic/Sports","Hospital","Clinic/Farm","Hospital/Clinic","Lab/Hospital",
        "Rehab Centre","School/Hospital","Eye Clinic","Ambulance/ER","Govt/NGO","MedTech Company"
    ]
}

ARTS_CAREERS = {
    "Career": [
        "Lawyer / Advocate","Corporate Lawyer","Judge","Journalist","News Anchor",
        "Content Creator","Graphic Designer","UX/UI Designer","Fashion Designer","Interior Designer",
        "Animator / VFX Artist","Filmmaker / Director","Photographer","Psychologist","Social Worker",
        "Teacher / Educator","Event Manager","Hotel Manager","Sports Coach","Political Scientist",
        "Museum Curator","Translator / Interpreter","Public Relations Officer","Advertising Executive"
    ],
    "Avg Salary (LPA)": [8,20,15,6,10,15,7,14,8,9,9,12,5,10,4,6,7,9,6,10,5,7,7,10],
    "Growth Outlook": [
        "Good","Excellent","Excellent","Moderate","Good","Excellent","Good","Excellent",
        "Good","Good","Good","Good","Moderate","Excellent","Good","Good","Good","Good","Good","Good",
        "Moderate","Good","Good","Good"
    ],
    "Key Exam": [
        "CLAT/AILET","CLAT/AILET","UPSC/State Judiciary","IIMC/CUET","Portfolio/Audition",
        "None","NID/NIFT/Portfolio","NID/Portfolio","NIFT/NID","NATA/NIFT",
        "FTII/Portfolio","FTII/SRFTI","Portfolio","RCI/NET","TISS/State",
        "CTET/TET","NIEM/Hospitality","NCHMCT JEE","SAI/NIS","UPSC/NET",
        "NET/State","State/Central","Portfolio/MBA","MBA/Portfolio"
    ]
}

SCIENCE_RESEARCH = {
    "Field": [
        "Astrophysicist / Astronomer","Biotechnologist","Forensic Scientist","Environmental Scientist",
        "Geologist","Oceanographer","Nuclear Scientist (BARC)","ISRO Scientist",
        "DRDO Scientist","Microbiologist","Nanotechnology Scientist","Wildlife Biologist",
        "Agricultural Scientist (ICAR)","Meteorologist / Weather Scientist","Seismologist"
    ],
    "Avg Salary (LPA)": [12,8,7,7,7,8,15,16,15,7,12,6,8,9,9],
    "Top Institute": [
        "IISc/IUCAA/TIFR","IIT/JNU","LNJN NCA/Forensic Science Univ","JNU/IIT Delhi",
        "IIT Roorkee/BHU","NIO Goa/Cochin Univ","IIT/IISc → BARC","IIT/NIT/IISc → ISRO",
        "IIT/NIT/IISc → DRDO","JNU/Delhi Univ","IIT Bombay/Delhi","WII Dehradun/JNU",
        "PAU/IARI/State Agri Univ","Pune Univ/IITM","IIT Roorkee/ISM"
    ],
    "Exam": [
        "JEST/GATE/NET","GATE/NET","GATE/State PSC","GATE/NET","GATE/GSI","GATE/NET",
        "GATE → OCES/BARC","GATE → ISRO Centralized","GATE → DRDO CEPTAM","CSIR-NET/GATE",
        "GATE/NET","GATE/Forest Service","ICAR AICE/NET","GATE/IMD exam","GATE/NET"
    ]
}

# ─────────────────────────────────────────────────────────────
# SECTION 6 │ CSS + UI HELPERS
# ─────────────────────────────────────────────────────────────
def render_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    html,body,[class*="css"]{font-family:'Outfit',sans-serif}
    .hero{background:linear-gradient(135deg,#0a0a1a 0%,#0d1b4b 50%,#1a0533 100%);
          border:1px solid rgba(99,120,255,.3);border-radius:20px;padding:40px 30px;
          text-align:center;margin-bottom:24px;}
    .hero h1{color:#fff;font-size:2.2rem;font-weight:800;margin:0 0 6px}
    .hero p{color:#94a3b8;font-size:1rem;margin:3px 0}
    .badge{display:inline-block;background:rgba(99,120,255,.18);border:1px solid rgba(99,120,255,.4);
           color:#a5b4fc;padding:3px 12px;border-radius:20px;font-size:.76rem;margin:3px}
    .card{border-radius:14px;padding:20px 22px;margin:10px 0;border:1.5px solid}
    .sec-title{font-size:1.2rem;font-weight:700;color:#e2e8f0;margin:20px 0 10px;
               border-left:4px solid #6378ff;padding-left:10px}
    .stButton>button{background:linear-gradient(135deg,#6378ff,#9f54d4);color:white;
                     border:none;padding:12px 0;border-radius:10px;font-size:1rem;
                     font-weight:700;width:100%;transition:all .3s}
    .stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(99,120,255,.45)}
    .opp-item{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.1);
              border-radius:10px;padding:9px 13px;margin:5px 0}
    </style>""", unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div class="hero">
        <h1>🎓 Career Compass India</h1>
        <p>AI-Powered Career Guidance for Students After 10th, 12th & Engineering</p>
        <p style="color:#6378ff;font-weight:600">150+ Careers · Govt Jobs · Engineering · Medical · Law · Defence</p>
        <div style="margin-top:12px">
            <span class="badge">🤖 TF-IDF + Cosine Similarity</span>
            <span class="badge">🏛️ UPSC · SSC · Banking · Railway</span>
            <span class="badge">🔬 JEE · NEET · GATE · NDA</span>
        </div>
    </div>""", unsafe_allow_html=True)

def display_stream_result(result):
    stream = result["stream"]
    cfg    = result["config"]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{cfg['bg']}55,{cfg['bg']}22);
                border:2px solid {cfg['color']};border-radius:18px;
                padding:28px;margin:16px 0;text-align:center">
        <div style="font-size:3rem">{cfg['emoji']}</div>
        <h2 style="color:{cfg['color']};margin:8px 0;font-size:1.8rem">
            Recommended Stream: {stream}
        </h2>
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Why this stream?")
        for r in cfg["reasons"]: st.markdown(f"• {r}")
        st.markdown(f"\n**🎓 Key Exams:** `{cfg['exams']}`")
    with c2:
        st.markdown("#### 🚀 Career Options:")
        st.info(cfg["careers"])
    st.markdown("#### 📊 Stream Suitability Scores:")
    score_df = pd.DataFrame({"Stream":list(result["scores"].keys()),
                             "Score":[round(v,3) for v in result["scores"].values()]}).set_index("Stream")
    st.bar_chart(score_df)
    st.caption("Score = Marks analysis (0–6 pts) + TF-IDF Cosine Similarity on interests (0–3 pts)")
    st.markdown('<div class="sec-title">🌟 Opportunities After 10th</div>', unsafe_allow_html=True)
    for section, items in OPPORTUNITIES_10TH.items():
        with st.expander(section):
            for name, desc in items:
                st.markdown(f"""<div class="opp-item"><strong>{name}</strong><br>
                    <span style="color:#94a3b8;font-size:.9rem">{desc}</span></div>""",
                    unsafe_allow_html=True)

def display_career_results(results_df):
    """Display top career recommendations — FIXED label and design."""
    if results_df.empty:
        st.error("❌ No recommendations found. Try entering more specific skills or interests.")
        return
    medals = ["🥇","🥈","🥉"]; colors = ["#FFD700","#C0C0C0","#CD7F32"]
    borders = ["#b8960c","#8a8a8a","#8b5e3c"]
    st.markdown(f'<div class="sec-title">🏆 Your Top Career Recommendations</div>', unsafe_allow_html=True)
    for idx, row in results_df.iterrows():
        score_pct = min(round(row["similarity_score"]*100, 1), 99.9)
        medal  = medals[idx] if idx < 3 else "⭐"
        color  = colors[idx] if idx < 3 else "#94a3b8"
        border = borders[idx] if idx < 3 else "#555"
        st.markdown(f"""
        <div style="border:2px solid {border};border-radius:16px;padding:20px;
                    margin:12px 0;background:rgba(255,255,255,.025)">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
                <h3 style="margin:0;color:{color};font-size:1.3rem">
                    {medal} #{idx+1} — {row['career']}
                </h3>
                <span style="background:{color};color:#111;padding:4px 14px;
                             border-radius:20px;font-weight:700;font-size:.88rem">
                    Match: {score_pct}%
                </span>
            </div>
            <p style="color:#cbd5e1;margin:7px 0 3px">{row['description']}</p>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"🏷️ **Category**\n\n`{row.get('category','—')}`")
            st.markdown(f"💰 **Salary**\n\n`{row['salary_range']}`")
        with c2:
            emoji = {"Excellent":"🚀","Good":"📈","Moderate":"➡️"}.get(row["growth_outlook"],"📊")
            st.markdown(f"{emoji} **Growth**\n\n**{row['growth_outlook']}**")
            st.markdown(f"📚 **After**\n\n`{row.get('after_class','12th')}`")
        with c3:
            st.markdown(f"📝 **Exam**\n\n`{row.get('exam_required','—')}`")
            st.markdown(f"🏫 **Top Colleges**\n\n`{row.get('top_colleges','—')}`")
        st.markdown(f"🛠️ **Skills:** `{row['skills'].title()}`")
        st.progress(min(row["similarity_score"], 1.0), text=f"Similarity Score: {score_pct}%")
        st.divider()

def show_opp_12th():
    st.markdown('<div class="sec-title">🌟 All Opportunities After 12th</div>', unsafe_allow_html=True)
    for section, items in OPPORTUNITIES_12TH.items():
        with st.expander(section, expanded=False):
            for name, desc in items:
                st.markdown(f"""<div class="opp-item"><strong>{name}</strong><br>
                    <span style="color:#94a3b8;font-size:.9rem">{desc}</span></div>""",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SECTION 7 │ DOMAIN CHARTS
# ─────────────────────────────────────────────────────────────
def show_engineering_chart():
    st.markdown('<div class="sec-title">⚙️ All Engineering Branches — Salary & Demand</div>',
                unsafe_allow_html=True)
    df_eng = pd.DataFrame(ENGINEERING_BRANCHES)
    st.info("🎓 **Eligibility:** 12th PCM | **Key Exam:** JEE Main, JEE Advanced, State CETs")
    view = st.radio("Sort by:", ["Job Demand (%)", "Avg Salary (LPA)"], horizontal=True, key="eng_sort")
    df_sorted = df_eng.sort_values(view, ascending=False)
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("##### 📊 Job Demand by Branch")
        chart_data = df_sorted.set_index("Branch")[["Job Demand (%)"]]
        st.bar_chart(chart_data, height=480, use_container_width=True, color="#00b4d8")
    with col2:
        st.markdown("##### 💰 Average Salary (LPA)")
        chart_data2 = df_sorted.set_index("Branch")[["Avg Salary (LPA)"]]
        st.bar_chart(chart_data2, height=480, use_container_width=True, color="#2dc653")
    st.markdown("##### 📋 Complete Engineering Branch Details")
    display_cols = df_eng[["Branch","Avg Salary (LPA)","Job Demand (%)","Top Exam","Key Recruiter"]]
    st.dataframe(display_cols.sort_values("Job Demand (%)", ascending=False),
                 use_container_width=True, hide_index=True)
    st.caption("💡 CS & AI have highest demand. Aerospace & Petroleum have highest salary potential.")

def show_medical_chart():
    st.markdown('<div class="sec-title">🏥 All Medical Careers — Salary & Study Duration</div>',
                unsafe_allow_html=True)
    df_med = pd.DataFrame(MEDICAL_CAREERS)
    st.info("🎓 **Eligibility:** 12th PCB | **Key Exam:** NEET-UG (single exam for all medical courses)")
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("##### 💰 Average Salary (LPA)")
        st.bar_chart(df_med.set_index("Career")[["Avg Salary (LPA)"]], height=420,
                     use_container_width=True, color="#e07a5f")
    with col2:
        st.markdown("##### ⏳ Study Duration (Years)")
        st.bar_chart(df_med.set_index("Career")[["Study Duration (Yrs)"]], height=420,
                     use_container_width=True, color="#9f54d4")
    st.markdown("##### 📋 Complete Medical Career Details")
    st.dataframe(df_med[["Career","Avg Salary (LPA)","Study Duration (Yrs)","Exam","Job Setting"]],
                 use_container_width=True, hide_index=True)
    st.caption("💡 Surgeons and Radiologists have highest salary. Nursing & MLT have fastest entry.")

def show_science_research_chart():
    st.markdown('<div class="sec-title">🔭 Science Research Careers — Salary & Institutes</div>',
                unsafe_allow_html=True)
    df_sci = pd.DataFrame(SCIENCE_RESEARCH)
    st.info("🎓 **Eligibility:** 12th PCM/PCB + B.Sc/B.Tech | **Key Exams:** GATE, CSIR-NET, JEST")
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("##### 💰 Average Salary by Research Field (LPA)")
        st.bar_chart(df_sci.set_index("Field")[["Avg Salary (LPA)"]], height=400,
                     use_container_width=True, color="#6378ff")
    with col2:
        st.markdown("##### 📋 Field Details")
        st.dataframe(df_sci[["Field","Avg Salary (LPA)","Exam"]],
                     use_container_width=True, height=400, hide_index=True)
    st.markdown("##### 🏫 Top Institutes for Each Research Field")
    for _, row in df_sci.iterrows():
        with st.expander(f"🔬 {row['Field']}"):
            st.markdown(f"**Top Institute:** {row['Top Institute']}")
            st.markdown(f"**Exam Required:** `{row['Exam']}`")
            st.markdown(f"**Average Salary:** ₹{row['Avg Salary (LPA)']}L per annum")

def show_arts_chart():
    st.markdown('<div class="sec-title">🎨 Arts & Humanities Careers — Salary & Growth</div>',
                unsafe_allow_html=True)
    df_arts = pd.DataFrame(ARTS_CAREERS)
    st.info("🎓 **Streams:** Arts/Humanities/Commerce | **Exams:** CLAT, NIFT, NID, IIMC, CTET, UPSC")
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("##### 💰 Average Salary (LPA)")
        df_sorted = df_arts.sort_values("Avg Salary (LPA)", ascending=False)
        st.bar_chart(df_sorted.set_index("Career")[["Avg Salary (LPA)"]], height=450,
                     use_container_width=True, color="#e07a5f")
    with col2:
        growth_counts = df_arts["Growth Outlook"].value_counts()
        st.markdown("##### 📊 Growth Outlook Distribution")
        st.bar_chart(growth_counts, height=200, use_container_width=True, color="#FFD700")
        st.markdown("##### 📋 Arts Careers — Full List")
        st.dataframe(df_arts[["Career","Avg Salary (LPA)","Growth Outlook","Key Exam"]],
                     use_container_width=True, height=220, hide_index=True)
    st.caption("💡 Corporate Lawyer, UX Designer & Judge have highest earning potential in Arts.")

def show_govt_jobs_page(df, vectorizer, tfidf_matrix):
    st.markdown('<div class="sec-title">🏛️ Government Jobs in India — Complete Guide</div>',
                unsafe_allow_html=True)
    st.info("💡 Govt jobs offer job security, pension, housing allowance, medical benefits, and social respect. "
            "India has lakhs of vacancies every year.")
    for cat, info in GOVT_JOB_CATEGORIES.items():
        with st.expander(cat, expanded=False):
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"**{info['desc']}**")
                st.markdown("**Key Exams:**")
                for exam in info["exams"]:
                    st.markdown(f"  • `{exam}`")
                st.markdown(f"**Eligibility:** {info['eligibility']}")
                st.markdown(f"**Prep Time:** {info['prep_time']}")
            with c2:
                st.metric("Salary Range", info["salary"])
    st.markdown('<div class="sec-title">🔍 Find Govt Jobs by Your Interests</div>',
                unsafe_allow_html=True)
    govt_interest = st.text_input("Your interests/skills:",
        placeholder="e.g. leadership, maths, law, nature, teaching, finance...")
    if st.button("🔍 Find Govt Jobs for Me", use_container_width=True, key="govt_btn"):
        if govt_interest.strip():
            results = get_recommendations(govt_interest, df, vectorizer, tfidf_matrix,
                                          category_filter="Government", top_n=5)
            if not results.empty:
                st.success(f"✅ Found {len(results)} government career matches!")
                display_career_results(results)
            else:
                st.warning("No specific matches. Try: 'leadership', 'law', 'finance', 'science'.")
        else:
            st.warning("Please enter your interests first.")

# ─────────────────────────────────────────────────────────────
# SECTION 8 │ MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Career Compass India", page_icon="🎓",
                       layout="wide", initial_sidebar_state="collapsed")
    render_css()
    hero()

    df                      = load_dataset()
    vectorizer, tfidf_matrix = build_tfidf_model(df)

    tabs = st.tabs([
        "🎯 Career Finder",
        "🏛️ Govt Jobs",
        "⚙️ Engineering Chart",
        "🏥 Medical Chart",
        "🔭 Science Chart",
        "🎨 Arts Chart",
        "📚 All Opportunities",
        "🔍 Explore Careers"
    ])

    # ── TAB 1: CAREER FINDER ──────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="sec-title">Step 1 — Select Your Level</div>', unsafe_allow_html=True)
        grade = st.radio("I have passed:", ["10th Grade", "12th Grade"], horizontal=True)
        st.divider()

        if grade == "10th Grade":
            st.markdown('<div class="sec-title">Step 2 — Enter Your Marks & Interests</div>',
                        unsafe_allow_html=True)
            c1,c2,c3,c4 = st.columns(4)
            with c1: math    = st.slider("📐 Mathematics (%)", 0, 100, 72)
            with c2: science = st.slider("🔬 Science (%)", 0, 100, 68)
            with c3: english = st.slider("📖 English (%)", 0, 100, 70)
            with c4: social  = st.slider("🗺️ Social Science (%)", 0, 100, 65)
            ca, cb = st.columns(2)
            with ca:
                interests = st.text_area("🌟 Interests",
                    placeholder="e.g. computers, drawing, biology, business, cooking, sports...", height=110)
            with cb:
                skills = st.text_area("🛠️ Skills",
                    placeholder="e.g. problem solving, leadership, creativity, coding...", height=110)
            if st.button("🎯 Recommend My Stream + Opportunities", use_container_width=True, key="btn10"):
                if not interests.strip() and not skills.strip():
                    st.warning("⚠️ Enter at least interests or skills for a better result.")
                else:
                    with st.spinner("Analysing your profile..."):
                        result = recommend_stream_for_10th(math, science, english, social, interests, skills)
                    st.success("✅ Analysis Complete!")
                    display_stream_result(result)

        else:
            st.markdown('<div class="sec-title">Step 2 — Enter Your Details</div>', unsafe_allow_html=True)
            col_a, col_b = st.columns([1,2])
            with col_a:
                stream = st.selectbox("🏫 Your 12th Stream", ["Science","Commerce","Arts"])
                category_filter = st.selectbox("🏷️ Career Category (optional)",
                    ["All Categories","Engineering","Medical","Government","Finance",
                     "Defense","Law","Management","Design","Media","Education",
                     "Health","Science","Hospitality","Sports","IT/Management"])
                # ── FIXED: label + range changed to 1–100 ──────────────
                top_n = st.slider("🎯 Top Results to Show", min_value=1, max_value=100,
                                  value=5, step=1,
                                  help="Set how many career recommendations you want to see")
                st.caption(f"Showing top **{top_n}** results")

            with col_b:
                interests = st.text_area("🌟 Interests",
                    placeholder="Science: coding, medicine, robotics...\n"
                                "Commerce: finance, marketing, business...\n"
                                "Arts: writing, design, law, psychology...", height=110)
                skills = st.text_area("🛠️ Skills",
                    placeholder="e.g. maths, programming, communication, creativity...", height=70)
                extra = st.text_input("🔍 Specific keyword (optional)",
                    placeholder="e.g. government, AI, environment, healthcare, sports...")

            if st.button("🚀 Find My Careers + Opportunities", use_container_width=True, key="btn12"):
                if not interests.strip() and not skills.strip():
                    st.error("❌ Please enter your interests and/or skills.")
                else:
                    full_profile = f"{interests} {skills} {extra}".strip()
                    with st.spinner("Running ML model..."):
                        results = get_recommendations(
                            user_text=full_profile, df=df,
                            vectorizer=vectorizer, tfidf_matrix=tfidf_matrix,
                            stream_filter=stream,
                            category_filter=category_filter if category_filter != "All Categories" else None,
                            top_n=top_n)
                    st.success(f"✅ Done! Showing top {len(results)} career matches.")
                    display_career_results(results)

                    with st.expander("🤖 How the ML Model Worked"):
                        n_stream = len(df[df["stream"]==stream])
                        st.markdown(f"""
                        **Your Profile:** *"{full_profile[:200]}..."*
                        
                        1. **TF-IDF Vectorization** — profile converted to a 5000-feature numerical vector
                        2. **Cosine Similarity** — compared against {n_stream} careers in {stream} stream  
                        3. **Ranking** — top {top_n} careers with highest scores returned
                        
                        *Score: 0% = no match, 100% = perfect match*
                        """)
                    show_opp_12th()

    # ── TAB 2: GOVT JOBS ────────────────────────────────────────────
    with tabs[1]:
        show_govt_jobs_page(df, vectorizer, tfidf_matrix)

    # ── TAB 3: ENGINEERING CHART ────────────────────────────────────
    with tabs[2]:
        show_engineering_chart()

    # ── TAB 4: MEDICAL CHART ────────────────────────────────────────
    with tabs[3]:
        show_medical_chart()

    # ── TAB 5: SCIENCE RESEARCH CHART ───────────────────────────────
    with tabs[4]:
        show_science_research_chart()

    # ── TAB 6: ARTS CHART ───────────────────────────────────────────
    with tabs[5]:
        show_arts_chart()

    # ── TAB 7: ALL OPPORTUNITIES ────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="sec-title">📚 Complete Opportunities Guide</div>', unsafe_allow_html=True)
        level = st.radio("Select Level:", ["After 10th","After 12th"], horizontal=True)
        source = OPPORTUNITIES_10TH if level == "After 10th" else OPPORTUNITIES_12TH
        for section, items in source.items():
            with st.expander(section, expanded=False):
                for name, desc in items:
                    st.markdown(f"""<div class="opp-item"><strong>{name}</strong><br>
                        <span style="color:#94a3b8;font-size:.9rem">{desc}</span></div>""",
                        unsafe_allow_html=True)

    # ── TAB 8: EXPLORE ALL CAREERS ──────────────────────────────────
    with tabs[7]:
        st.markdown('<div class="sec-title">🔍 Explore All 150+ Careers</div>', unsafe_allow_html=True)
        fc1,fc2,fc3 = st.columns(3)
        with fc1: search_q = st.text_input("🔍 Search career / skill:", "")
        with fc2: f_stream  = st.selectbox("Stream", ["All","Science","Commerce","Arts"])
        with fc3:
            all_cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
            f_cat = st.selectbox("Category", all_cats)
        filtered = df.copy()
        if search_q:
            mask = (filtered["career"].str.contains(search_q,case=False,na=False) |
                    filtered["skills"].str.contains(search_q,case=False,na=False) |
                    filtered["description"].str.contains(search_q,case=False,na=False))
            filtered = filtered[mask]
        if f_stream != "All":    filtered = filtered[filtered["stream"] == f_stream]
        if f_cat != "All":       filtered = filtered[filtered["category"] == f_cat]
        st.caption(f"Showing {len(filtered)} of {len(df)} careers")
        st.dataframe(filtered[["career","category","stream","after_class","salary_range",
                                "growth_outlook","exam_required"]].rename(columns={
            "career":"Career","category":"Category","stream":"Stream",
            "after_class":"After Class","salary_range":"Salary",
            "growth_outlook":"Growth","exam_required":"Exam Required"}),
            use_container_width=True, height=480)
        sel = st.selectbox("📋 View Full Details:", ["— Select —"] + filtered["career"].tolist())
        if sel != "— Select —":
            row = df[df["career"]==sel].iloc[0]
            st.markdown(f"### {row['career']}")
            st.markdown(f"**Category:** `{row['category']}` | **Stream:** `{row['stream']}` | **After:** `{row['after_class']}`")
            st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Skills Required:** `{row['skills'].title()}`")
            c1,c2,c3 = st.columns(3)
            c1.metric("Salary Range", row["salary_range"])
            c2.metric("Growth Outlook", row["growth_outlook"])
            c3.metric("Exam Required", row["exam_required"])
            st.markdown(f"**Top Colleges:** {row['top_colleges']}")

    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#475569;font-size:.82rem;padding:14px 0">
        🎓 Career Compass India v3.0 | 150+ Careers · Govt Jobs · Engineering · Medical · Law · Defence<br>
        Built with Python · Streamlit · Scikit-learn (TF-IDF + Cosine Similarity)<br>
        <em>Mini Project — Final Year ML Project</em>
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()