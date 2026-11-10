import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("1. Reading creditcard.csv...")
df = pd.read_csv("creditcard.csv")

# Extract all fraud cases (492) + sample 500 legitimate cases
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0].sample(n=500, random_state=42)

# Merge & shuffle
small_df = pd.concat([fraud, normal]).sample(frac=1, random_state=42)

# Select key features to keep UI inputs clean
features = ['V1', 'V2', 'V3', 'V4', 'V14', 'Amount']
X = small_df[features]
y = small_df['Class']

# Train Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Save model binary
joblib.dump(model, "fraud_model.pkl")
print("✅ Saved 'fraud_model.pkl' successfully!")