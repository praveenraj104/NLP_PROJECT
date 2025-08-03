#  Fake News Detection using XAI (LIME)

A machine learning-based web application that detects whether a news article is **fake or real**, and explains the reasoning behind the prediction using **LIME (Local Interpretable Model-agnostic Explanations)** — a powerful eXplainable AI tool.

---

##  Project Overview

With the rise of misinformation, automated systems for fake news detection have become critical. However, most models act like black boxes. This project focuses not only on **accuracy**, but also on **explainability**, so users can understand *why* a particular news article is flagged as fake or real.

---

##  Core Features

-  **Fake/Real Classification** of news articles using NLP and ML
-  **Explainability with LIME** – Highlights the most influential words behind each prediction
-  **Model trained on trusted datasets** like FakeNewsNet or Kaggle’s Fake News Dataset
-  **Simple and clean web interface** for entering and testing news articles
-  **View and interpret model decisions** in an easy-to-understand format

---

##  Tech Stack

| Component       | Technology                    |
|-----------------|-------------------------------|
| Backend         | Python (Flask)                |
| NLP & ML        | Scikit-learn, NLTK / SpaCy    |
| Explainability  | LIME (TextExplainer)          |
| Frontend        | HTML, CSS, JavaScript         |
| Dataset         | Fake News Dataset (Kaggle or similar) |

---

##  How It Works

1. User submits a news article or headline.
2. The model vectorizes the input and predicts whether it's real or fake.
3. LIME analyzes the prediction and highlights the **key words** that influenced the decision.
4. The user sees the result along with a **visual explanation** from LIME.

---



