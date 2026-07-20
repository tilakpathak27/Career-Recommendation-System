# 🎓 Career Recommendation System

An AI-powered career guidance tool for students after **10th and 12th grade**, built using Python, Streamlit, and Machine Learning (TF-IDF + Cosine Similarity).

---

## 📸 Features

| Feature | Description |
|---|---|
| 🎯 10th Grade | Recommends the best stream (Science / Commerce / Arts) |
| 🚀 12th Grade | Recommends Top 3 careers based on stream, skills & interests |
| 🤖 ML Model | TF-IDF Vectorization + Cosine Similarity |
| 📊 Dataset | 50+ curated careers across all streams |
| 📈 Similarity Score | Shows how well each career matches your profile |
| 💡 Career Details | Description, salary range, growth outlook, required skills |
| ✅ Input Validation | Handles empty or invalid inputs gracefully |

---

## 🗂️ Project Structure

```
career-recommendation-system/
│
├── app.py                  # Main Streamlit app (all code here)
├── career_dataset.csv      # Dataset with 50+ careers
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ How to Run the Project

### Step 1 — Install Python
Make sure you have **Python 3.8 or higher** installed.
Download from: https://www.python.org/downloads/

### Step 2 — Install Dependencies
Open your terminal / command prompt and run:

```bash
pip install streamlit pandas numpy scikit-learn
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3 — Run the App
Make sure `app.py` and `career_dataset.csv` are in the **same folder**, then run:

```bash
streamlit run app.py
```

The app will open automatically in your browser at: **http://localhost:8501**

---

## 🧠 How the ML Works

### TF-IDF (Term Frequency – Inverse Document Frequency)
- Converts skill text into numerical vectors
- Gives higher weight to unique/important skills
- Ignores common words that appear in all careers

### Cosine Similarity
- Measures the "angle" between the user's profile vector and each career vector
- Score ranges from 0 (no match) to 1 (perfect match)
- Top 3 highest scores = Top 3 career recommendations

---

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
```

---

## 🧪 Sample Test Cases

### For 10th Grade:
- Math: 85%, Science: 80%, English: 70%
- Interests: "computers, robotics, coding"
- Skills: "problem solving, mathematics"
- **Expected:** Science stream ✅

### For 12th Grade (Science):
- Interests: "data analysis, statistics, artificial intelligence"
- Skills: "programming, mathematics, Python"
- **Expected:** Data Scientist / AI Engineer ✅

---

## 📌 Notes for Students

- This project can be extended with a **neural network** for better accuracy
- You can add more careers to `career_dataset.csv` without changing any code
- The ML model re-trains instantly when you add new data

---

## 🏫 Credits
Built as a Final Year Machine Learning Project.
Technology Stack: Python · Streamlit · Scikit-learn · Pandas · NumPy