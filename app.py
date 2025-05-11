from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from lime.lime_text import LimeTextExplainer

# Initialize Flask app
app = Flask(__name__)

# Load the tokenizer and model from the local directory
tokenizer = BertTokenizer.from_pretrained('./model')
model = BertForSequenceClassification.from_pretrained('./model')

# LIME Text Explainer with corrected class names
explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
  # class 0 = FAKE, class 1 = REAL

# Prediction function
def predict_fake_news(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    confidence = torch.max(probs).item()
    prediction = torch.argmax(probs).item()

    # ✅ FIXED: correct label mapping based on model's class order
    label = 'FAKE' if prediction == 1 else 'REAL'

    
    return label, confidence, probs.tolist()[0], text

# LIME explanation function
def explain_prediction(text):
    def predict_proba(texts):
        inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.softmax(outputs.logits, dim=1).cpu().numpy()

    explanation = explainer.explain_instance(text, predict_proba, num_features=10)
    return explanation

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Form-based prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('news_text', '')
    if text:
        label, confidence, probs, original_text = predict_fake_news(text)
        explanation = explain_prediction(text)
        explanation_html = explanation.as_html()
        return render_template('index.html', 
                               prediction=label, 
                               confidence=round(confidence * 100, 2),
                               probs=[round(p * 100, 2) for p in probs],
                               explanation=explanation_html)
    return render_template('index.html', prediction="No input provided.")

# JSON API route
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get('text', '')
    label, confidence, probs, _ = predict_fake_news(text)
    explanation = explain_prediction(text)
    explanation_html = explanation.as_html()
    return jsonify({
        'prediction': label,
        'confidence': round(confidence * 100, 2),
        'probabilities': [round(p * 100, 2) for p in probs],
        'explanation': explanation_html
    })

# Optional test prediction
if __name__ == '__main__':
    # Test prediction to verify
    test_text = "Earth will experience 6 days of darkness."
label, confidence, probs, _ = predict_fake_news(test_text)
print("🧪 TEST PREDICTION ON SAMPLE TEXT")
print(f"Text: {test_text}")
print(f"Prediction: {label}")
print(f"Confidence: {round(confidence * 100, 2)} %")
print(f"Class Probabilities: [REAL, FAKE] -> {[round(p * 100, 2) for p in probs]}")

    
app.run(debug=True)
