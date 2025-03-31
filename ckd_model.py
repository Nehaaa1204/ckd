# ckd_model.py

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from boruta import BorutaPy

# Load dataset
df = pd.read_csv('kidney_disease.csv')

# Preprocess
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df[col] = df[col].fillna(df[col].median())

# Feature/target split
X = df.drop(['classification', 'id'], axis=1)
y = df['classification']

# Feature selection
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=1, random_state=42)
boruta_selector.fit(X.values, y.values)

# Get selected features
selected_features = X.columns[boruta_selector.support_].to_list()
X_selected = X[selected_features]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save the trained model & scaler
joblib.dump(knn, 'ckd_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'selected_features.pkl')

print("Model training completed and saved successfully.")
