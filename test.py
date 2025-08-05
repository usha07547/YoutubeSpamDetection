# Import required libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Check if dataset exists
dataset_path = 'D:/FlaskApp/WebApp/Youtube-Spam-Dataset.csv'
print(dataset_path)
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Load dataset
df = pd.read_csv(dataset_path)
#print(df.head())
# Drop rows with missing values (optional)
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ...existing code...

# Split features and label
if 'CLASS' not in df.columns:
    raise ValueError("Target column 'CLASS' not found in dataset.")
X = df.drop('CLASS', axis=1)
y = df['CLASS']

# ...existing code...

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Model Accuracy:", accuracy)
