import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("D:\\My Work\\Rainfall\\rainfall.csv")

# Fill missing values with column mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode SUBDIVISION
le = LabelEncoder()
df['SUBDIVISION_ENC'] = le.fit_transform(df['SUBDIVISION'])

# Scale YEAR feature
scaler = StandardScaler()
df['YEAR_SCALED'] = scaler.fit_transform(df[['YEAR']])

# Prepare monthly data
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

records = []
for _, row in df.iterrows():
    for i, month in enumerate(months):
        records.append({
            'SUBDIVISION_ENC': row['SUBDIVISION_ENC'],
            'YEAR': row['YEAR'],
            'MONTH_NUM': i + 1,
            'RAINFALL': row[month]
        })
monthly_df = pd.DataFrame(records)

# Split based on year
train_df = monthly_df[monthly_df['YEAR'] <= 2005]
test_df = monthly_df[monthly_df['YEAR'] > 2005]

features = ['SUBDIVISION_ENC', 'YEAR', 'MONTH_NUM']
X_train = train_df[features]
y_train = train_df['RAINFALL']
X_test = test_df[features]
y_test = test_df['RAINFALL']

# Train models
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Save models and encoders
joblib.dump(lr, "linear_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "year_scaler.pkl")

# Evaluate models
def evaluate(model, name):
    preds = model.predict(X_test)
    print(f"\n{name} Evaluation")
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R²:", r2_score(y_test, preds))

evaluate(lr, "Linear Regression")
evaluate(rf, "Random Forest")