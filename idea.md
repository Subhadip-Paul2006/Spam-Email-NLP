Spam Email Detector with NLP*

*🎯 Project Goal*
Build an AI app that reads any email text and tells you if it’s *Spam* or *Ham* in 1 second.

Spam = unwanted/promotional mail. Ham = legit mail like “Team meeting at 4 PM”.

*🧠 Skills You’ll Learn*

*Python:* Strings, Functions, Lists

*Data:* Pandas, NumPy for dataset handling

*NLP:* Text cleaning, Tokenization, Stopwords, Stemming, TF-IDF

*ML:* Naive Bayes, Logistic Regression, Random Forest

*Deployment:* Streamlit for web app

*📂 Step 1: Dataset*
Typical format: 2 columns
Email Text | Label

"Win ₹1 Lakh now, click here" | 1 → Spam
"Project report attached" | 0 → Ham

Label: 0 = Ham, 1 = Spam

*📊 Step 2: Load & Explore Data*
import pandas as pd
df = pd.read_csv("spam.csv")
print(df.head())
print(df['label'].value_counts())

Check: total emails, spam %, missing values.

*🧹 Step 3: Text Preprocessing*

Raw: `"Congratulations!!! You WON ₹50,000... CLICK NOW!!!"`

Clean: `"congratulation won click"`

*Convert to lowercase*
text = text.lower()

*Remove punctuation*
import string
text = text.translate(str.maketrans('', '', string.punctuation))

*Step 4: Tokenization*
Split sentence into words

`"you won prize"` → `["you", "won", "prize"]`

from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)

*Step 5: Remove Stopwords*

Remove common words: 
the, is, a, an
`["you", "won", "the", "prize"]` → `["won", "prize"]`

from nltk.corpus import stopwords
words = [w for w in tokens if w not in stopwords.words('english')]

*Step 6: Stemming*

Reduce words to root form

`running, runs, ran` → `run`
`playing, played` → `play`

from nltk.stem import PorterStemmer
ps = PorterStemmer()
words = [ps.stem(w) for w in words]

*Step 7: Convert Text to Numbers with TF-IDF*

ML models can’t read text. TF-IDF gives importance score to words.

Spam words like “win, free, offer” get high scores.

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])

*Step 8: Train Model*
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

*Other models to try:* Multinomial Naive Bayes, Random Forest, XGBoost

*Step 9: Predict New Email*

sample = ["Congratulations you won free iPhone"]

sample_vec = tfidf.transform(sample)
pred = model.predict(sample_vec)
print("Spam" if pred[0]==1 else "Ham")

*Step 10: Check Performance*
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))

Also check Precision, Recall, F1-Score, Confusion Matrix.

*🎨 Step 11: Build Streamlit App*
import streamlit as st
st.title("Spam Email Detector")
email = st.text_area("Paste email text here")
if st.button("Check"):
    email_vec = tfidf.transform([email])
    result = model.predict(email_vec)
    if result[0]==1:
        st.error("🚨 Spam Email Detected")
    else:
        st.success("✅ Legit Email")

Run: `streamlit run app.py`

*⭐ Features to Add*

*Beginner:* Show accuracy, simple UI

*Intermediate:* Add spam probability %, save email history

*Advanced:* Multi-language support, Phishing detection, Gmail API integration

*📁 Project Folder Structure*
spam-detector/
├── data/spam.csv
├── models/model.pkl
├── notebooks/training.ipynb
├── app.py
├── train.py
├── requirements.txt
└── README.md

*💼 Resume Bullet*
_Spam Email Classifier using NLP_
Built end-to-end text classification pipeline with Python, NLTK, TF-IDF, and Logistic Regression. Achieved 97%+ accuracy. Deployed interactive Streamlit app for real-time spam detection.

*🚀 Mini Challenge for You*
1. Add phishing email detection
2. Show spam probability percentage
3. Deploy on Hugging Face Spaces for free
