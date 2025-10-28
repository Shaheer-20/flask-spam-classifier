import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- App and Database Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key'
# Sets the database URI. The '///' means a relative path from the 'instance' folder.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///classifier.db'
db = SQLAlchemy(app)

# --- Database Model Definition ---
class Email(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(10), nullable=False)
    # The actual_label is what the user confirms. It can be null at first.
    actual_label = db.Column(db.String(10), nullable=True)
    verified = db.Column(db.Boolean, default=False, nullable=False)

# --- Model Loading ---
def load_model_and_vectorizer():
    try:
        model = joblib.load('model/spam_classifier_model.pkl')
        vectorizer = joblib.load('model/vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        # If no model exists, create a default one from the initial CSV
        print("Model not found. Training a new one from data/spam.csv...")
        initial_train()
        return joblib.load('model/spam_classifier_model.pkl'), joblib.load('model/vectorizer.pkl')

def initial_train():
    """Trains and saves an initial model from spam.csv."""
    df = pd.read_csv('data/spam.csv')
    vectorizer = CountVectorizer()
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
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predicts the label for an email and stores it in the database."""
    email_text = request.form.get('email_text')
    if not email_text:
        return render_template('index.html')

    email_counts = vectorizer.transform([email_text])
    probabilities = model.predict_proba(email_counts)[0]
    prediction = model.classes_[probabilities.argmax()]
    confidence = round(probabilities.max() * 100, 2)

    # Save the prediction to the database for later review
    new_email = Email(text=email_text, predicted_label=prediction)
    db.session.add(new_email)
    db.session.commit()

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        email_text=email_text
    )

@app.route('/review')
def review():
    """Shows all unverified emails for user feedback."""
    unverified_emails = Email.query.filter_by(verified=False).all()
    return render_template('review.html', emails=unverified_emails)

@app.route('/verify/<int:email_id>/<string:correct_label>')
def verify(email_id, correct_label):
    """Marks an email as verified with the correct label."""
    email = Email.query.get_or_404(email_id)
    email.actual_label = correct_label
    email.verified = True
    db.session.commit()
    flash(f'Email marked as {correct_label}.', 'success')
    return redirect(url_for('review'))

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrains the model using all verified data from the database."""
    global model, vectorizer # We need to update the global model objects

    verified_emails = Email.query.filter_by(verified=True).all()

    if len(verified_emails) < 10: # Set a reasonable minimum for retraining
        flash('Not enough verified emails to retrain. Please verify at least 10 emails.', 'error')
        return redirect(url_for('review'))

    # Create a DataFrame from the database data
    data = {'text': [email.text for email in verified_emails],
            'label': [email.actual_label for email in verified_emails]}
    df = pd.DataFrame(data)

    # Retrain vectorizer and model
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(df['text'])
    model = MultinomialNB()
    model.fit(X_counts, df['label'])

    # Save the new model
    joblib.dump(model, 'model/spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'model/vectorizer.pkl')

    flash(f'Model successfully retrained with {len(verified_emails)} emails!', 'success')
    return redirect(url_for('review'))

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        # This will create the database file and table if they don't exist
        db.create_all()
    app.run(debug=True)