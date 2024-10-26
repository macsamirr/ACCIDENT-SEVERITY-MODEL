import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('accident.xlsx')

# Check for missing values
if df.isnull().sum().any():
    print("Warning: Missing values found in the dataset.")

# Map target variable (Accident Severity) to numerical values
severity_mapping = {'Minor': 1, 'Serious': 2, 'Fatal': 3}
df['Accident Severity'] = df['Accident Severity'].map(severity_mapping)

# Convert 'Alcohol' column to numeric values
df['Alcohol'] = df['Alcohol'].map({'No': 0, 'Yes': 1})


df = pd.get_dummies(df, columns=['Weather', 'Road Surface', 'Time of Day', 'Collision Type'], drop_first=True)


X = df.drop(columns=['Accident Severity'])
y = df['Accident Severity']

# Convert boolean columns to integers
X = X.astype(int)

# Check the first few rows and data types of features
print("Feature Data Types:\n", X.dtypes)
print("First few rows of the features:\n", X.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the types of the features in X_train
print("X_train Data Types:\n", X_train.dtypes)

# Identify non-numeric columns
non_numeric_columns = X_train.select_dtypes(exclude=['int64', 'float64']).columns
if len(non_numeric_columns) > 0:
    print("Non-numeric columns in X_train:", non_numeric_columns.tolist())

   
    for col in non_numeric_columns:
        print(f"Unique values in {col}: {X_train[col].unique()}")

    raise ValueError("X_train contains non-numeric values. Please check your data.")


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'accident_severity_rf_model.pkl')
joblib.dump(scaler, 'scaler_rf.pkl')


y_pred = model.predict(X_test)
print(f"Model R^2 score: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
