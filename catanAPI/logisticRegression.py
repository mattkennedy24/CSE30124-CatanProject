# Create train and test data sets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dfCatan = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

#Let's use production, tradeGain, robberCardsGain and tradeLoss only
inputData = dfCatan[['production','tradeGain','robberCardsGain','tradeLoss']]

# Create a new column called win that is equal to one if the row's points value >= 10
dfCatan['win'] = (dfCatan['points'] >= 10).astype(int)
print(dfCatan['win'])
targetVal = dfCatan['win']

#And scale the data to have mean=0 and std=1
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(inputData)
inputData_scaled = scaler.transform(inputData)

X_train, X_test, y_train, y_test = train_test_split(inputData_scaled, targetVal, test_size=0.20, random_state=42)


# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)