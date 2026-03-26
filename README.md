# 🎓 Student Performance Predictor

A machine learning web app that predicts whether a student will **Pass or Fail** based on their academic habits — built with Python, Flask, and a Random Forest Classifier.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen?style=flat-square)

---

## 🖼️ Features

- **ML-Powered Prediction** — Random Forest Classifier trained on 1,000 student records
- **Interactive UI** — Clean sliders and toggles for all input parameters
- **Confidence Score** — Pass/Fail probability breakdown shown as progress bars
- **Personalized Tips** — AI-generated improvement suggestions based on your inputs
- **REST API** — `/predict` endpoint returns JSON, easily extensible

---

## 📊 Input Features

| Feature | Description | Range |
|---|---|---|
| Study Hours / Day | Hours spent studying daily | 0 – 10 |
| Attendance | Class attendance percentage | 40% – 100% |
| Previous Score | Score in last exam | 30 – 100 |
| Assignments Done | % of assignments submitted | 0% – 100% |
| Sleep Hours | Hours of sleep per night | 3 – 10 |
| Extra-Curricular | Participates in activities | Yes / No |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Flask |
| ML Model | Random Forest (scikit-learn) |
| Data Processing | pandas, numpy |
| Model Persistence | pickle |
| Frontend | HTML5, CSS3, Vanilla JS |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (generates `model.pkl`)
```bash
python train_model.py
```

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
student-performance-predictor/
├── app.py              # Flask web application
├── train_model.py      # Data generation + model training
├── model.pkl           # Trained Random Forest model
├── student_data.csv    # Generated synthetic dataset
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Frontend (HTML + CSS + JS)
└── README.md
```

---

## 🔌 API Usage

**POST** `/predict`

```json
// Request
{
  "study_hours": 5.0,
  "attendance": 80,
  "prev_score": 65,
  "assignments_done": 75,
  "sleep_hours": 7.0,
  "extra_curricular": 1
}

// Response
{
  "prediction": "Pass",
  "confidence": 87.3,
  "pass_prob": 87.3,
  "fail_prob": 12.7,
  "tips": ["🌟 Great profile! Keep maintaining your study habits."]
}
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **88%** |
| Precision (Pass) | 0.88 |
| Recall (Pass) | 0.88 |
| F1-Score | 0.88 |

Top feature by importance: **Study Hours** (48%) → **Previous Score** (21%) → **Assignments Done** (11%)

---

## 👨‍💻 Author

**Ayesha **  
B.Sc. IT Student | Aspiring AI Engineer  
[GitHub](https://github.com/ayeshasiddiqua8881-bit) · [LinkedIn](www.linkedin.com/in/ayeshasiddiquaabdulkalam71ab7a391)

---

## 📄 License

MIT License — feel free to use, modify, and build on this project.
