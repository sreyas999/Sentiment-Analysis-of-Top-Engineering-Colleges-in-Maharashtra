# Sentiment-Analysis-of-Top-Engineering-Colleges-in-Maharashtra
This project is a full-stack web application built with Flask (Python) that performs sentiment analysis on student reviews of engineering colleges. It combines machine learning, data visualization, and user interaction to provide insights into how students perceive various institutions.
---

## 🚀 Features

✅ User signup, login, and password reset (with email verification)  
✅ Feedback submission and storage in SQLite  
✅ Sentiment analysis of reviews using pre-trained ML models  
✅ Pie chart visualization of positive, negative, and neutral reviews  
✅ Secure forms with CSRF protection  
✅ Responsive front-end with dynamic college data display  
✅ Admin-friendly database structure for easy scalability  

---

## 🧠 Tech Stack

**Backend:** Flask (Python)  
**Frontend:** HTML5, CSS3, JavaScript, Bootstrap  
**Database:** SQLite  
**Machine Learning:** Scikit-learn, Pandas, Pickle  
**Email Service:** Flask-Mail  
**Visualization:** Matplotlib  

---

## 🧩 Machine Learning Models Used

- **SVM Model (`svm_model.pkl`)** — for sentiment classification  
- **Logistic Regression Model (`lr_model.pkl`)** — for cross-validation  
- **CountVectorizer (`vectorizer.pkl`)** — for text feature extraction  

---

## 🗂️ Folder Structure

project-folder/
│
├── app.py # Main Flask app file
├── forms.py # WTForms for login
├── requirements.txt # Python dependencies
├── Procfile # Render/Heroku deployment config
├── static/ # CSS, JS, Images
├── templates/ # HTML templates (Jinja2)
├── sentiment_result.csv # Dataset for sentiment analysis
├── svm_model.pkl
├── lr_model.pkl
├── vectorizer.pkl
└── flask_users.db # SQLite database

---

## ⚙️ Installation & Setup

1. Clone the repository  
   ```bash
   git clone https://github.com/sreyas999/college-review-system.git
   cd college-review-system
