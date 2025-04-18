# Spam Detection Web Application

## 📌 Overview

A simple web application that detects spam messages using Natural Language Processing (NLP) and machine learning. The system analyzes text input and classifies it as either "spam" or "not spam" (ham).

## 🛠️ How It Works

### 1. Data Processing

- **Dataset**: Uses SMS Spam Collection from Kaggle
- **Text Cleaning**:
  - Converts to lowercase
  - Removes special characters (keeps $, %, !, ?)
  - Replaces numbers with "NUM"
  - Stems words (e.g., "running" → "run")
  - Removes unimportant stopwords

### 2. Model Training

- **Algorithm**: Logistic Regression (simple but effective)
- **Key Features**:
  - Uses word combinations (1-3 words together)
  - Focuses on top 3000 important words
  - Automatically balances spam/ham classes
- **Training Process**:
  ```python
  1. Load and clean the data
  2. Create word features (TF-IDF vectors)
  3. Train classifier on these features
  4. Save model for later use
  ```

### 3. Prediction

- **How It Decides**:
  - Analyzes word patterns in your message
  - Checks for spam indicators ($, free, win, etc.)
  - Calculates probability of being spam
  - Uses dynamic threshold (0.25-0.35)

## 🚀 How to Run

### Requirements

- Python 3.8+
- Node.js
- Python packages: `scikit-learn`, `pandas`, `nltk`, `joblib`
- Node packages: `express`, `body-parser`

### Setup Steps

1. **Install Python dependencies**:

   ```bash
   cd model-training
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords')"
   ```

2. **Train the model**:

   ```bash
   python train_model.py
   ```

3. **Install Node dependencies**:

   ```bash
   npm install
   ```

4. **Run the web app**:

   ```bash
   node server.js
   ```

5. **Access the app**: Open `http://localhost:3000`

## 📂 Project Structure

```
spam-filter-app/
├── data/                   # Dataset folder
│   └── spam.csv            # Original dataset
├── models/                 # Trained model storage
├── model-training/         # Python training code
│   ├── train_model.py      # Model training script
│   └── predict.py          # Prediction script
├── public/                 # Frontend files
│   ├── index.html          # User interface
│   └── style.css           # Styling
├── server.js               # Node.js backend
└── package.json            # Node.js dependencies
```

## 🔍 Understanding the Model

### Why Logistic Regression?

- Simple to understand
- Works well with text data
- Provides probability scores
- Handles imbalanced data well

### Key Features That Detect Spam

1. **Money-related terms** ($, cash, prize)
2. **Urgency words** (urgent, immediately)
3. **Winning claims** (win, free, reward)
4. **Suspicious patterns** (click here, call now)

## ✨ Customization Tips

1. **Improve Accuracy**:

   - Add more spam examples to training data
   - Try different thresholds in `predict.py`

2. **Extend Features**:

   ```python
   # In train_model.py, you can add:
   df['has_links'] = df['text'].str.contains('http').astype(int)
   ```

3. **Test Different Models**:
   - Try `RandomForestClassifier` for better performance
   - Experiment with neural networks for advanced usage

## 📊 Sample Results

| Message                   | Prediction  |
| ------------------------- | ----------- |
| "Win a free iPhone!"      | Spam 🚨     |
| "Your package arrived"    | Not Spam ✅ |
| "Claim your $1000 prize"  | Spam 🚨     |
| "Meeting at 3pm tomorrow" | Not Spam ✅ |

This simple yet effective system gives you practical experience with NLP and web integration while maintaining easy understandability.
