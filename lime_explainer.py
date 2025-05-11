from lime.lime_text import LimeTextExplainer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('model/tokenizer')
model = BertForSequenceClassification.from_pretrained('model/model.safetensors')

# LIME Explainer setup
explainer = LimeTextExplainer(class_names=['Real', 'Fake'])

# Function to predict with LIME
def model_predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    return logits.detach().numpy()

def explain_prediction(text):
    explanation = explainer.explain_instance(text, model_predict, num_features=10)
    return explanation.as_html()
