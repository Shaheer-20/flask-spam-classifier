# Flask Email Spam Classifier 📧

A web-based application built with Flask and Scikit-learn that classifies emails as "Spam" or "Ham" (Not Spam). This project features an interactive feedback loop, allowing the user to correct the model's predictions and retrain it on the fly, continuously improving its accuracy.

## 🚀 Key Features

* **Naive Bayes Classifier**: Utilizes the Multinomial Naive Bayes algorithm for fast and effective text classification.
* **Prediction Confidence**: Displays a confidence score for each prediction, showing how certain the model is.
* **Interactive UI**: A clean and simple user interface for testing new emails.
* **Database Integration**: Stores every prediction in an SQLite database for review.
* **Feedback Loop**: A dedicated review page where users can correct the model's mistakes.
* **Live Retraining**: A one-click button to retrain the model using all the user-verified data, improving its intelligence over time.

---

## 🛠️ Technologies Used

* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn
* **Database**: SQLite with Flask-SQLAlchemy
* **Frontend**: HTML, CSS
* **Deployment**: (Ready for deployment on platforms like Heroku or PythonAnywhere)

---

## ⚙️ Setup and Installation

Follow these steps to get the application running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/flask-spam-classifier.git](https://github.com/your-username/flask-spam-classifier.git)
cd flask-spam-classifier