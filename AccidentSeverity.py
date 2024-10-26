import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('/content/road_accident_data.csv')

# Map target variable (Accident Severity) to numerical values
severity_mapping = {'Minor': 1, 'Serious': 2, 'Fatal': 3}
df['Accident Severity'] = df['Accident Severity'].map(severity_mapping)

# Convert 'Alcohol' column to numeric values
df['Alcohol'] = df['Alcohol'].map({'No': 0, 'Yes': 1})

# Preprocess categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Weather', 'Road Surface', 'Time of Day', 'Collision Type'], drop_first=True)

# Define independent variables (features) and dependent variable (target)
X = df.drop(columns=['Accident Severity'])
y = df['Accident Severity']

# Convert boolean columns to integers
X = X.astype(int)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the saved Random Forest model
model = joblib.load('accident_severity_rf_model.pkl')

# Predict accident severity using the entire dataset
predictions = model.predict(X_scaled)

# Add predictions to the DataFrame
df['Predicted Severity'] = predictions

# Print the original and predicted severity
print(df[['Accident Severity', 'Predicted Severity']].head())
