# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the data
data = pd.read_csv('task1_dataset-insurance.csv')

# Data check
print(data.head())
print(data.describe())
print(data['sex'].unique())
print(data['smoker'].unique())
print(data['region'].unique())

#Check for null values
print(data.isna().sum())

# Encode categorical variables
label_encoder = LabelEncoder()

data['sex'] = label_encoder.fit_transform(data['sex'])
data['smoker'] = label_encoder.fit_transform(data['smoker'])
data['region'] = label_encoder.fit_transform(data['region'])

# One-hot encode categorical variables
one_hot_encoder = OneHotEncoder()
data_encoded = one_hot_encoder.fit_transform(data[['region']]).toarray()

# Combine the one-hot encoded data with the original data
data = pd.concat([data, pd.DataFrame(data_encoded, columns=['region_0', 'region_1', 'region_2', 'region_3'])], axis=1)

# Split the data
x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Print the coefficients
coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:")
for feature, coef in zip(x.columns, coefficients):
    print(f"{feature}: {coef}")

print(f"Intercept: {intercept}")