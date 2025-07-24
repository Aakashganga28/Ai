from flask import Flask, render_template, request
import joblib
import seaborn as sb
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords once if needed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load saved model and vectorizers
model = joblib.load('rf_model.pkl')
tfidf = joblib.load('tfidf.pkl')
countV = joblib.load('countvectorizer.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    tokens = nltk.word_tokenize(review)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

def fake_news_det(news):
    cleaned_text = preprocess_text(news)
    count_input = countV.transform([cleaned_text])
    tfidf_input = tfidf.transform(count_input)
    prediction = model.predict(tfidf_input)
    confidence = model.predict_proba(tfidf_input)

    result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
    conf_score = f"Confidence → Fake: {confidence[0][0]:.2f}, Real: {confidence[0][1]:.2f}"
    return result, conf_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred, confidence = fake_news_det(message)
        return render_template('index.html', prediction=pred, confidence=confidence)
    return render_template('index.html', prediction="❌ Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)
