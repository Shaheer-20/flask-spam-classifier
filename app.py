import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- App and Database Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///classifier.db'
db = SQLAlchemy(app)

# --- Database Model Definition ---
class Email(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(10), nullable=False)
    actual_label = db.Column(db.String(10), nullable=True)
    verified = db.Column(db.Boolean, default=False, nullable=False)

# --- Model Loading ---
def load_model_and_vectorizer():
    try:
        model = joblib.load('model/spam_classifier_model.pkl')
        vectorizer = joblib.load('model/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        print("Model not found. Training a new one from data/spam.csv...")
        initial_train()
        return joblib.load('model/spam_classifier_model.pkl'), joblib.load('model/vectorizer.pkl')

def initial_train():
    df = pd.read_csv('data/spam.csv')
    vectorizer = CountVectorizer(stop_words='english')
    X_counts = vectorizer.fit_transform(df['text'])
    model = MultinomialNB()
    model.fit(X_counts, df['label'])
    joblib.dump(model, 'model/spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')
    print("Initial model trained and saved.")

model, vectorizer = load_model_and_vectorizer()

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_text')
    if not email_text:
        return render_template('index.html')

    email_counts = vectorizer.transform([email_text])
    probabilities = model.predict_proba(email_counts)[0]
    prediction = model.classes_[probabilities.argmax()]
    confidence = round(probabilities.max() * 100, 2)

    new_email = Email(text=email_text, predicted_label=prediction)
    db.session.add(new_email)
    db.session.commit()

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'prediction': prediction, 'confidence': confidence})

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        email_text=email_text
    )

@app.route('/review')
def review():
    unverified_emails = Email.query.filter_by(verified=False).all()
    return render_template('review.html', emails=unverified_emails)

@app.route('/verify/<int:email_id>/<string:correct_label>')
def verify(email_id, correct_label):
    email = Email.query.get_or_404(email_id)
    email.actual_label = correct_label
    email.verified = True
    db.session.commit()
    flash(f'Email marked as {correct_label}.', 'success')
    return redirect(url_for('review'))

@app.route('/retrain', methods=['POST'])
def retrain():
    global model, vectorizer
    model_choice = request.form.get('model_choice', 'naive_bayes')

    verified_emails = Email.query.filter_by(verified=True).all()

    if len(verified_emails) < 10:
        flash('Not enough verified emails to retrain. Please verify at least 10 emails.', 'error')
        return redirect(url_for('review'))

    data = {'text': [email.text for email in verified_emails],
            'label': [email.actual_label for email in verified_emails]}
    df = pd.DataFrame(data)

    if model_choice in ['logistic_regression', 'svm']:
        vectorizer = TfidfVectorizer(stop_words='english')
        if model_choice == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
        else:
            model = LinearSVC(max_iter=1000, dual=True)
    else:
        vectorizer = CountVectorizer(stop_words='english')
        model = MultinomialNB()

    X_transformed = vectorizer.fit_transform(df['text'])
    model.fit(X_transformed, df['label'])

    joblib.dump(model, 'model/spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

    flash(f'Model successfully retrained using {model.__class__.__name__}!', 'success')
    return redirect(url_for('review'))

@app.route('/dashboard')
def dashboard():
    verified_emails = Email.query.filter_by(verified=True).all()

    if len(verified_emails) < 2:
        flash('Not enough verified data to build a dashboard. Please verify more emails.', 'error')
        return redirect(url_for('review'))

    true_labels = [email.actual_label for email in verified_emails]
    texts = [email.text for email in verified_emails]
    X_counts = vectorizer.transform(texts)
    predictions = model.predict(X_counts)

    accuracy = accuracy_score(true_labels, predictions)
    try:
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        cm = confusion_matrix(true_labels, predictions, labels=model.classes_).tolist()
    except ValueError:
        flash('Could not generate report. Ensure you have verified examples of both ham and spam.', 'error')
        return redirect(url_for('review'))

    return render_template(
        'dashboard.html',
        accuracy=accuracy,
        report=report,
        confusion_matrix=cm,
        labels=model.classes_.tolist()
    )

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)