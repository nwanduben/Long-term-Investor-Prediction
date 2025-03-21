from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Select only top 5 features
top_features = ["duration", "balance", "age", "campaign", "pdays"]
X_train_selected = X_train[top_features]
X_val_selected = X_val[top_features]

# Train Random Forest on top 5 features
rf_model_small = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_small.fit(X_train_selected, y_train)

# Evaluate performance
y_pred = rf_model_small.predict(X_val_selected)
accuracy = accuracy_score(y_val, y_pred)

print(f"✅ Accuracy with only top 5 features: {accuracy:.2f}")
