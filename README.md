
# 🔮 Customer Churn Prediction Model

A Machine Learning project that predicts customer churn using Logistic Regression. This project demonstrates the end-to-end ML pipeline from data generation to model evaluation.

## 🎯 Objective
To predict whether a customer will leave the service (churn) based on their demographics, usage patterns, and contract details. This is a classic classification problem in business analytics.

## 🧠 Methodology
1.  **Data Simulation**: Generates a synthetic dataset mimicking telecom customer data (Age, Tenure, Monthly Charge, etc.).
2.  **Preprocessing**: Handles categorical variables (Contract Type) and scales numerical features.
3.  **Model Training**: Trains a Logistic Regression model using `scikit-learn`.
4.  **Evaluation**: Assesses model performance using Accuracy, Confusion Matrix, and Classification Report.
5.  **Visualization**: Plots the Confusion Matrix as a heatmap.

## 🛠️ Tech Stack
- **Python 3.x**
- **Scikit-learn** (Machine Learning)
- **Pandas & NumPy** (Data Handling)
- **Seaborn & Matplotlib** (Visualization)

## 📦 Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/sololeveling-01/churn-prediction.git
    cd churn-prediction
    ```
2.  Install dependencies:
    ```bash
    pip install scikit-learn pandas numpy matplotlib seaborn
    ```

## 🏃‍♂️ Usage
Run the prediction script:
```bash
python churn_prediction.py
```
This will print the model's accuracy and classification report to the console and save `confusion_matrix.png`.

## 📊 Results
The model achieves an accuracy of ~70-80% on the synthetic test set, demonstrating the effectiveness of logistic regression for this type of binary classification problem.
