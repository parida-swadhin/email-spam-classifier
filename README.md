# 📧 Email Spam Classification System

> A Machine Learning powered NLP application that classifies emails as **Spam** or **Ham** using TF-IDF and Naive Bayes.

---

##  Project Overview

This project builds an end-to-end **Email Spam Detection System** using Natural Language Processing (NLP) techniques and a supervised Machine Learning model.

The system analyzes raw email text, converts it into numerical features using **TF-IDF Vectorization**, and classifies it using a **Multinomial Naive Bayes algorithm**.

The model is deployed using **Streamlit**, allowing real-time spam prediction through a simple web interface.

---

## 🎯 Problem Statement

Spam emails:

- Reduce productivity  
- Waste storage resources  
- Increase security risks  

The objective of this project is to design a machine learning pipeline that can automatically detect and filter spam emails efficiently and accurately.

---

##  Solution Architecture

### 1️ Data Preprocessing
- Convert text to lowercase
- Remove punctuation and special characters
- Remove stopwords
- Tokenization

### 2️ Feature Engineering
- TF-IDF (Term Frequency – Inverse Document Frequency)
- Converts text into meaningful numerical vectors

### 3️ Model Training
- Multinomial Naive Bayes
- Trained on labeled dataset
- Model saved using Pickle

### 4️ Deployment
- Streamlit-based Web Application
- Accepts user input
- Predicts Spam / Ham instantly

---

## 🛠 Tech Stack

- **Language:** Python  
- **Data Handling:** Pandas  
- **NLP:** NLTK  
- **Feature Extraction:** Scikit-learn (TF-IDF)  
- **Model:** Naive Bayes  
- **Deployment:** Streamlit  
- **Model Saving:** Pickle  

---
## 📂 Project Structure

```
email-spam-classifier/
│
├── data/
│   └── spam.csv
│
├── model/
│   └── spam_model.pkl
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Model Evaluation

The model performance is evaluated using:

- ✅ Accuracy  
- ✅ Precision  
- ✅ Recall  
- ✅ F1-Score  
- ✅ Confusion Matrix  

These metrics ensure:
- Spam emails are correctly identified  
- False positives are minimized  

---
##  Installation & Setup

---
### 1️ Clone the Repository
```bash
git clone  https://github.com/parida-swadhin/email-spam-classifier.git
cd email-spam-classifier
```

---
### 2️ Install Dependencies
```bash
pip install -r requirements.txt
```

---
### 3️ Train the Model
```bash
python train.py
```

---
### 4️ Run the Application
```bash
streamlit run app.py
```
---

##  Application Preview

The web interface allows users to:

- Enter email text  
- Click **Predict**  
- Instantly view whether the message is Spam or Ham  

(You can add screenshots here later.)

---

## 📌 Key Highlights

- ✔ Modular project structure  
- ✔ End-to-end ML pipeline  
- ✔ Real-time prediction  
- ✔ Clean and scalable implementation  
- ✔ Beginner-friendly architecture  

---

##  Real-World Applications

- Email Filtering Systems  
- SMS Spam Detection  
- Enterprise Email Security  
- Customer Support Automation  

---

##  Future Improvements

- Implement Logistic Regression / SVM  
- Use Deep Learning models (LSTM / Transformers)  
- Deploy on AWS / Render  
- Convert into REST API  
- Add model performance visualization dashboard  

---
##  Author

**Parida Swadhin**  
Aspiring Machine Learning Engineer  
Focused on building practical AI applications and strengthening ML fundamentals.
