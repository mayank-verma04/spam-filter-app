# predict.py
import sys
import joblib

model = joblib.load('models/spam_model.pkl')
text = sys.argv[1]

# Get confidence score
spam_prob = model.predict_proba([text])[0][1]

# Dynamic thresholding
threshold = 0.25 if any(word in text.lower() for word in ['win', 'free', '$']) else 0.35
print('spam' if spam_prob > threshold else 'ham')