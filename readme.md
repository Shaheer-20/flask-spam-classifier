# Flask Email Spam Classifier 📧

A dynamic web application built with Flask and Scikit-learn that classifies emails as **Spam** or **Ham** (Not Spam). This project features a powerful interactive feedback loop, allowing users to correct the model's predictions and retrain it on the fly, continuously improving its accuracy over time.

---

## ✨ Key Features

-   **Intelligent Classification**: Utilizes a Multinomial Naive Bayes classifier for fast and effective text analysis.
-   **Confidence Score**: Displays the model's confidence level for each prediction, offering greater insight.
-   **Continuous Learning**: Every email classified is stored in a database, creating an ever-growing dataset.
-   **Interactive Feedback Loop**: A dedicated "Review" page allows you to correct the model's mistakes, turning it into a powerful teaching tool.
-   **One-Click Retraining**: Retrain the model instantly with all the verified data you've provided, making it smarter with a single click.
-   **Modern UI**: A clean, responsive, and user-friendly interface for a smooth experience.



---

## 🛠️ Tech Stack

-   **Backend**: Python, Flask, Flask-SQLAlchemy
-   **Machine Learning**: Scikit-learn, Pandas
-   **Database**: SQLite
-   **Frontend**: HTML, CSS

---

## ⚙️ Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.8+
-   Git

### 1. Clone the Repository

```bash
git clone [https://github.com/Shaheer-20/flask-spam-classifier.git](https://github.com/Shaheer-20/flask-spam-classifier.git)
cd flask-spam-classifier