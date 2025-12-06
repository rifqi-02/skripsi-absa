# ABSA Streamlit App — LDA + SVM for Female Daily Reviews

This repository contains a Streamlit-based application for **Aspect-Based Sentiment Analysis (ABSA)** applied to product reviews from **Female Daily**.  
The system uses **Latent Dirichlet Allocation (LDA)** for aspect extraction and **Support Vector Machine (SVM)** for sentiment classification at the segment level.

---

## 🚀 Features

### 🔹 1. Single Review Analysis
- Automatic text segmentation  
- Aspect detection using LDA (5 aspects: Kemasan, Aroma, Tekstur, Harga, Efek)  
- Sentiment classification (Positive/Negative) using SVM  
- Clean and interactive UI

### 🔹 2. Full Dataset Dashboard
- Upload CSV/Excel Female Daily dataset  
- Automatic ABSA for all reviews  
- Visualizations:
  - Sentiment distribution per aspect  
  - Skin type vs sentiment  
  - Age group vs sentiment  
  - WordCloud positive/negative segments  
- Segment-level result table

---

## 🧠 Model Components

### **Aspect Extraction**
- Gensim LDA model
- Bigram phrase model
- Seed-based aspect boosting
- Dictionary + topic–aspect mapping

### **Sentiment Classification**
- One SVM model per aspect
- TF-IDF vectorizers for each aspect
- Preprocessing and token normalization

