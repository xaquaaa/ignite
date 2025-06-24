import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load realistic dataset with noise
df = pd.read_csv("survey_numerical_realistic_noisy.csv")

# Define features and target
X = df.iloc[:, :-1]  # All questions as features
y = df.iloc[:, -1]   # Target variable (learning style)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train optimized RandomForest model with better generalization
model = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
