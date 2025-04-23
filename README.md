# 📧 Spam Message Classifier

A machine learning-based web application that classifies SMS messages as **Spam** or **Ham (Not Spam)**. It uses Natural Language Processing (NLP) techniques and various classification models to achieve high accuracy and precision.

---

## 🔍 Project Overview

This project demonstrates the complete machine learning pipeline for text classification:

1. **Data Cleaning**
2. **Exploratory Data Analysis (EDA)**
3. **Text Preprocessing**
4. **Model Building & Evaluation**
5. **Model Improvement with Ensemble Techniques**
6. **Web Application Interface**
7. **Deployment**

---

## 📊 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Format**: CSV file with labeled SMS messages.
- **Columns Used**:
  - `v1` → `target` (spam/ham)
  - `v2` → `text` (SMS content)

---

## 🔧 Technologies Used

- **Python** for core logic
- **Scikit-learn** for modeling
- **NLTK** for text preprocessing
- **Matplotlib / Seaborn** for visualization
- **TfidfVectorizer** for feature extraction
- **Flask / Streamlit** (for web UI)
- **Pickle** for saving models
- **XGBoost** for boosting performance

---

## 🧹 Data Preprocessing Steps

- Lowercasing
- Tokenization
- Removing special characters, punctuation
- Removing stopwords
- Stemming

---

## 🧠 Models Used

| Algorithm        | Description                    |
|------------------|--------------------------------|
| Naive Bayes      | Probabilistic text classifier  |
| SVM              | Support Vector Machine         |
| Logistic Regression | Linear classifier             |
| Decision Tree    | Tree-based method              |
| Random Forest    | Ensemble of decision trees     |
| KNN              | Instance-based classifier      |
| AdaBoost         | Boosted weak learners          |
| Extra Trees      | Ensemble random decision trees |
| Gradient Boost   | Boosting technique             |
| XGBoost          | Extreme Gradient Boosting      |
| Voting & Stacking | Ensemble meta-learners         |

---

## 📈 Performance Comparison

Models were evaluated using:
- **Accuracy**
- **Precision**
- **Confusion Matrix**

Visualizations included:
- Pie charts for spam/ham distribution
- Histograms of character/word/sentence count
- WordClouds for frequent terms

---

## 💡 Final Model Selection

- **Model**: `Multinomial Naive Bayes`  
- **Vectorizer**: `TF-IDF` with 3000 features  
- **Ensembles**: Voting & Stacking for improved precision

---

## 🚀 Deployment

To use the model in a web app:

1. **Load Pretrained Files**:
   ```python
   import pickle
   tf = pickle.load(open('vectorizer.pkl', 'rb'))
   model = pickle.load(open('model.pkl', 'rb'))
