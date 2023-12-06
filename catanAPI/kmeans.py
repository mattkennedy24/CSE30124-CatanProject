import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


dfCatanStats = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

dfCatanStats['win'] = (dfCatanStats['points'] >= 10).astype(int)

# Drop irrelevant columns for clustering
dfCatanStats = dfCatanStats.drop(columns=['gameNum', 'points', 'player', 'me', 'settlement1', 'Unnamed: 16', 'Unnamed: 17',
       'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'settlement2',
       'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
       'Unnamed: 26'])

print(dfCatanStats)
cluster_cols = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'production', 'tradeGain', 'robberCardsGain', 'totalGain', 'tradeLoss', 'robberCardsLoss', 'tribute', 'totalLoss', 'totalAvailable']
df_cluster = dfCatanStats[cluster_cols]

# Standardize the data
scaler = StandardScaler()
df_cluster_scaled = scaler.fit_transform(df_cluster)

num_clusters = 20

# Apply K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
dfCatanStats['cluster'] = kmeans.fit_predict(df_cluster_scaled)

# Define features and target variable
features = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'production', 'tradeGain', 'robberCardsGain', 'totalGain', 'tradeLoss', 'robberCardsLoss', 'tribute', 'totalLoss', 'totalAvailable', 'cluster']
target = 'win'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dfCatanStats[features], dfCatanStats[target], test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Get feature importances
feature_importances = clf.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort features by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances')
plt.savefig('feature_importances.png') 


