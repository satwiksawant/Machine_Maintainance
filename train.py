

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib  # For saving the model
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r"predictive_maintenance.csv")

# Print the column names to check for any discrepancies
print("Columns in the DataFrame:", df.columns.tolist())

# Convert columns to categorical types
df['Product ID'] = df['Product ID'].astype('category')
df['Type'] = df['Type'].astype('category')
df['Failure Type'] = df['Failure Type'].astype('category')

# Drop unnecessary columns
df.drop(columns=['UDI', 'Product ID', 'Target'], inplace=True)

# Correct the numerical and categorical columns as needed
num_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]']  # Adjusted to remove 'Temperature Difference' and any missing columns
cat_cols = ['Type']

# Split data into features and target
X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for categorical features
cat_transformer = Pipeline(steps=[
    ('label', OrdinalEncoder())
])

# Pipeline for numerical features
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine pipelines into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC()
}

# Train models and evaluate
best_model = None
best_accuracy = 0
for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print(f"Best Model: {best_model} with Accuracy: {best_accuracy:.2f}")

# Save the best model to a .pkl file
joblib.dump(best_model, 'models/best_model.pkl')
print("Best model saved as 'models/best_model.pkl'")
