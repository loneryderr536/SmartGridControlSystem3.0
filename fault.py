import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------
# Data Loading & Cleaning
# ----------------------------

# Load the dataset (update the path if necessary)
df = pd.read_csv('data/detect_dataset.csv')

# Remove unwanted columns (e.g., any columns with 'Unnamed' in their names)
cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
if cols_to_drop:
    df.drop(cols_to_drop, axis=1, inplace=True)

# ----------------------------
# Define Features & Target
# ----------------------------

# Features: Electrical measurements; Target: Fault (1) or No Fault (0)
features = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
target = 'Output (S)'

X = df[features]
y = df[target]

# ----------------------------
# Data Splitting & Scaling
# ----------------------------

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Although Random Forests are tree-based (and thus scale-invariant), we use scaling for consistency.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Train Random Forest Classifier
# ----------------------------

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ----------------------------
# Model Evaluation
# ----------------------------

y_pred = rf_model.predict(X_test_scaled)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fault', 'Fault'],
            yticklabels=['No Fault', 'Fault'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ----------------------------
# Save the Trained Model
# ----------------------------

import joblib
joblib.dump(rf_model, 'rf_fault_detection_model.pkl')
print("Random Forest model saved as 'rf_fault_detection_model.pkl'")
