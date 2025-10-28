import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# 1. Load Data
df = pd.read_csv('data/spam.csv')

# 2. Separate features (X) and target (y)
X = df['text']
y = df['label']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Extraction (Convert text to numerical data)
# We use CountVectorizer to convert email text into a matrix of token counts.
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 5. Train the Model
# We initialize and train the Multinomial Naive Bayes classifier.
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# 6. Evaluate the Model (Optional, but good practice)
accuracy = model.score(X_test_counts, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save the Model and the Vectorizer
# We need to save both so we can use them for new predictions in our app.
joblib.dump(model, 'model/spam_classifier_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Model and vectorizer saved successfully.")