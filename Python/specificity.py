#specificity tells us how well a model can recognize the negative class when it’s actually negative.


from sklearn.metrics import confusion_matrix

# Example true labels and predictions
y_true = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0]
y_pred = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate specificity
specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

print(f"Specificity: {specificity:.2f}")
