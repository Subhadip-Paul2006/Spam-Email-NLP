# 🎯 Spam Email Detector — NLP-Powered Email Classification

> **One-second AI-powered spam detection** — paste any email text and instantly know if it's **Spam** 🚨 or **Ham** ✅.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/NLTK-3.8%2B-green?logo=nltk" alt="NLTK">
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture Diagram](#-architecture-diagram)
- [C4 Model Diagrams](#-c4-model-diagrams)
- [Project Mindmap](#-project-mindmap)
- [Project Tree Structure](#-project-tree-structure)
- [Sprint Kanban](#-sprint-kanban)
- [Dataset Distribution](#-dataset-distribution)
- [Quick Start](#-quick-start)
- [Tech Stack](#-tech-stack)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

**Spam Email Detector** is an end-to-end machine learning application that classifies emails as **Spam** (unwanted/promotional) or **Ham** (legitimate) using Natural Language Processing (NLP) techniques. The project covers the full ML lifecycle — from data preprocessing to model training to interactive web deployment.

### What it does
| Input | Process | Output |
|-------|---------|--------|
| Raw email text | NLP pipeline (cleaning → tokenization → TF-IDF → ML model) | Spam 🚨 or Ham ✅ + confidence % |

---

## ⭐ Features

| Level | Feature | Status |
|-------|---------|--------|
| 🟢 Beginner | Real-time spam/ham classification | ✅ Done |
| 🟢 Beginner | Accuracy, precision, recall metrics | ✅ Done |
| 🟢 Beginner | Clean Streamlit UI | ✅ Done |
| 🟡 Intermediate | Spam probability percentage display | 🔄 Planned |
| 🟡 Intermediate | Email history logging | 🔄 Planned |
| 🔴 Advanced | Multi-language support | 📋 Future |
| 🔴 Advanced | Phishing email detection | 📋 Future |
| 🔴 Advanced | Gmail API integration | 📋 Future |

---

## 🏗️ Architecture Diagram

High-level system architecture showing how all components interact:

```mermaid
flowchart TB
    subgraph User["👤 User Layer"]
        A[User Browser]
    end

    subgraph Frontend["🎨 Frontend Layer"]
        B[Streamlit Web App<br/>app.py]
        C[Text Input Area]
        D[Result Display<br/>Spam/Ham + %]
    end

    subgraph Backend["⚙️ Backend Layer"]
        E[NLP Preprocessor]
        F[TF-IDF Vectorizer]
        G[ML Classifier]
        H[Model Serializer<br/>joblib]
    end

    subgraph Storage["💾 Storage Layer"]
        I[(spam.csv<br/>Dataset)]
        J[(model.pkl<br/>Trained Model)]
        K[(tfidf.pkl<br/>Vectorizer)]
        L[(history.db<br/>Email Log)]
    end

    subgraph External["🌐 External"]
        M[NLTK Corpora<br/>Stopwords/Stemmers]
        N[Scikit-learn<br/>Algorithms]
    end

    A --> B
    B --> C
    C --> E
    E --> F
    F --> G
    G --> D
    G --> H
    H --> J
    H --> K
    E -.-> M
    G -.-> N
    B --> L
    I -.-> E
    J -.-> G
```

---

## 📐 C4 Model Diagrams

### Level 1 — System Context

```mermaid
C4Context
    title System Context — Spam Email Detector

    Person(user, "Email User", "Anyone who wants to check if an email is spam")

    System(spd, "Spam Email Detector", "NLP-powered web application that classifies emails as Spam or Ham")

    System_Ext(nltk, "NLTK Library", "Natural Language Toolkit for text processing")
    System_Ext(dataset, "Email Dataset", "CSV file containing labeled email samples")
    System_Ext(browser, "Web Browser", "User's browser to access Streamlit app")

    Rel(user, browser, "Uses")
    Rel(browser, spd, "Interacts via HTTP")
    Rel(spd, dataset, "Reads training data")
    Rel(spd, nltk, "Uses for NLP processing")

    UpdateLayoutConfig($topMargin: 80, $leftMargin: 150)
```

### Level 2 — Container

```mermaid
C4Container
    title Container Diagram — Spam Email Detector

    Person(user, "Email User")

    Container(web, "Streamlit Web App", "Python/Streamlit", "Provides UI for email input and result display")
    Container(preproc, "NLP Pipeline", "Python/NLTK", "Cleans, tokenizes, stems, and vectorizes email text")
    Container(ml, "ML Model", "Python/scikit-learn", "Classifies vectorized text as Spam or Ham")
    ContainerDb(csv, "Dataset Store", "CSV File", "Stores raw email text and labels")
    ContainerDb(pkl, "Model Store", "Pickle Files", "Stores serialized model and vectorizer")

    Rel(user, web, "HTTP/8501")
    Rel(web, preproc, "Calls")
    Rel(preproc, ml, "Feeds TF-IDF vectors")
    Rel(ml, web, "Returns prediction")
    Rel(csv, preproc, "Training data")
    Rel(ml, pkl, "Load/Save")
    Rel(preproc, pkl, "Load/Save vectorizer")
```

### Level 3 — Component

```mermaid
C4Component
    title Component Diagram — Streamlit App

    Container(app, "Streamlit App", "app.py")

    Component(ui, "UI Module", "Text Area + Button", "Collects email input from user")
    Component(nlp, "NLP Processor", "preprocess.py", "Cleans and transforms text")
    Component(modelMgr, "Model Manager", "model_manager.py", "Loads model and vectorizer")
    Component(history, "History Logger", "history.py", "Saves classification results")
    Component(display, "Result Renderer", "app.py", "Shows spam/ham result with confidence")

    Rel(ui, nlp, "Raw text")
    Rel(nlp, modelMgr, "Cleaned text")
    Rel(modelMgr, display, "Prediction + %")
    Rel(display, history, "Log result")
```

---

## 🗺️ Project Mindmap

```mermaid
mindmap
  root((Spam Email
  Detector))
    Data
      Dataset (spam.csv)
      CSV format
      2 columns: text + label
      Label: 0=Ham, 1=Spam
      Exploration (Pandas)
    NLP Pipeline
      Lowercase conversion
      Punctuation removal
      Tokenization (NLTK)
      Stopword removal
      Stemming (Porter)
      TF-IDF Vectorization
    ML Models
      Logistic Regression
      Multinomial Naive Bayes
      Random Forest
      XGBoost
    Evaluation
      Accuracy
      Precision
      Recall
      F1-Score
      Confusion Matrix
    Deployment
      Streamlit Web App
      HuggingFace Spaces
      Docker Container
    Future Scope
      Multi-language
      Phishing Detection
      Gmail API
      Spam Probability %
      Email History Log
```

---

## 🌳 Project Tree Structure

```mermaid
tree
  root[spam-detector]
    data
      spam.csv
      raw/
        spam_assassin.csv
        enron.csv
    models
      model.pkl
      tfidf.pkl
    notebooks
      01_eda.ipynb
      02_training.ipynb
      03_evaluation.ipynb
    src
      __init__.py
      preprocess.py
      train.py
      predict.py
      model_manager.py
      history.py
    app.py
    requirements.txt
    README.md
    PRD.md
    TRD.md
    AI_WORKFLOW.md
    USAGE.md
    .gitignore
    LICENSE
```

> **Expanded file listing:**

```
spam-detector/
├── 📂 data/
│   ├── spam.csv                    # Primary dataset
│   └── 📂 raw/                     # Raw downloaded datasets
│       ├── spam_assassin.csv
│       └── enron.csv
├── 📂 models/
│   ├── model.pkl                   # Serialized trained model
│   └── tfidf.pkl                   # Serialized TF-IDF vectorizer
├── 📂 notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_training.ipynb           # Model training experiments
│   └── 03_evaluation.ipynb         # Evaluation & comparison
├── 📂 src/
│   ├── __init__.py
│   ├── preprocess.py               # NLP cleaning pipeline
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Prediction utilities
│   ├── model_manager.py            # Load/save model helpers
│   └── history.py                  # Email history logger
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── PRD.md                          # Product Requirements Document
├── TRD.md                          # Technical Requirements Document
├── AI_WORKFLOW.md                  # AI/ML workflow details
├── USAGE.md                        # Setup & usage guide
├── .gitignore                      # Git ignore rules
└── LICENSE                         # MIT License
```

---

## 📋 Sprint Kanban

```mermaid
kanban
  title Sprint Board — Spam Email Detector

  section Backlog
    Gmail API Integration
    Multi-language Support
    Phishing Detection
    Docker Deployment

  section To Do
    Email History Logging
    Spam Probability %
    Unit Tests (pytest)

  section In Progress
    Streamlit UI Polish
    Model Optimization

  section Done
    Dataset Collection
    NLP Pipeline
    Model Training
    Basic Web App
    README & Docs
```

---

## 📊 Dataset Distribution

```mermaid
pie title Email Dataset Distribution — Spam vs Ham
    "Ham (Legitimate)" : 4825
    "Spam (Unwanted)" : 1500
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector

# 2. Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Train the model
python src/train.py

# 6. Run the app
streamlit run app.py
```

> 📖 For detailed instructions, see [USAGE.md](./USAGE.md)

---

## 💻 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.10+ | Core development |
| Data | Pandas, NumPy | Data manipulation |
| NLP | NLTK, scikit-learn | Text processing & vectorization |
| ML | Logistic Regression, Naive Bayes, Random Forest | Classification |
| UI | Streamlit | Interactive web application |
| Serialization | joblib | Model persistence |
| Visualization | Matplotlib, Seaborn, WordCloud | EDA & charts |

---

## 📸 Screenshots

> *Add screenshots here after running the app:*

| Description | Image |
|-------------|-------|
| App Home Screen | ![Home](docs/images/home.png) |
| Spam Detection Result | ![Spam](docs/images/spam.png) |
| Ham Detection Result | ![Ham](docs/images/ham.png) |

---

## 🤝 Contributing

Contributions are welcome! Please read [USAGE.md](./USAGE.md) for the complete contribution guide.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

<p align="center">
  Built with ❤️ using Python, NLP & Machine Learning
</p>
