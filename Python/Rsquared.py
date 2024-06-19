import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Creating an artificial dataset
data = {
    'Temperature': [22, 21, 23, 24, 20, 19, 25, 28, 27, 26],
    'Humidity': [80, 82, 78, 76, 80, 85, 70, 68, 75, 72],
    'Rainfall': [0, 2, 0, 1, 0, 5, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

# Define independent and dependent variables
X = df[['Temperature', 'Humidity']]
y = df['Rainfall']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print(f'R-squared: {r2}')
