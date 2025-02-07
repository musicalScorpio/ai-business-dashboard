from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Task description
task_text = "Task started on 01/01/2025 and finished on 01/15/2025. The task was supposed to be completed by 01/10/2025."

# Labels for classification
labels = ["Positive", "Negative"]

# Make the classification
prediction = classifier(task_text, candidate_labels=labels)

# Print result
print(f"Prediction: {prediction['labels'][0]} (Confidence: {prediction['scores'][0]:.2f})")
