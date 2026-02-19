
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Generate dummy customer data
np.random.seed(42)
n_samples = 1000

data = {
    'Age': np.random.randint(18, 70, n_samples),
    'Tenure': np.random.randint(1, 48, n_samples), # Months as customer
    'MonthlyCharge': np.random.uniform(20, 100, n_samples),
    'Usage': np.random.randint(10, 500, n_samples), # GB usage? or meaningful metric
    'SupportCalls': np.random.randint(0, 5, n_samples),
    'ContractType': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 30% churn rate
}

df = pd.DataFrame(data)

# Preprocessing
# Convert categorical 'ContractType' to numerical
contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['ContractType'] = df['ContractType'].map(contract_map)

# Feature Selection
X = df[['Age', 'Tenure', 'MonthlyCharge', 'Usage', 'SupportCalls', 'ContractType']]
y = df['Churn']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Customer Churn Prediction')
plt.savefig('confusion_matrix.png')
print("\nSaved confusion_matrix.png")
