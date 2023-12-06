import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

dfCatan = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

# Create a new column called win that is equal to one if the row's points value >= 10
dfCatan['win'] = (dfCatan['points'] >= 10).astype(int)
dfCatan['me'].fillna(0, inplace = True)

dfCatan = dfCatan.rename(columns={'settlement1':'set1_num1', 'Unnamed: 16':'set1_res1', 'Unnamed: 17':'set1_num2', 'Unnamed: 18':'set1_res2', 'Unnamed: 19':'set1_num3', 'Unnamed: 20':'set1_res3', 'settlement2':'set2_num1', 'Unnamed: 22':'set2_res1', 'Unnamed: 23':'set2_num2', 'Unnamed: 24':'set2_res2', 'Unnamed: 25':'set2_num3', 'Unnamed: 26':'set2_res3'})

dfSettlement = dfCatan[['set1_num1', 'set1_res1', 'set1_num2', 'set1_res2', 'set1_num3', 'set1_res3',
                        'set2_num1', 'set2_res1', 'set2_num2', 'set2_res2', 'set2_num3', 'set2_res3',
                        'points', 'win', 'gameNum', 'production']]


# Print the columns of the new DataFrame
print("Cols in dfSettlement: ", dfSettlement.columns)

userBoard = {'tiles':
                [
                {'numberToken': '10', 'resource': 'S', 'location': 0, 'port': ''}, 
                {'numberToken': '11', 'resource': 'W', 'location': 1, 'port': '3G'},
                {'numberToken': '11', 'resource': 'S', 'location': 2, 'port': ''},
                {'numberToken': '3', 'resource': 'L', 'location': 3, 'port': '2S'},
                {'numberToken': '9', 'resource': 'W', 'location': 4, 'port': ''}, 
                {'numberToken': '4', 'resource': 'L', 'location': 5, 'port': ''}, 
                {'numberToken': '5', 'resource': 'S', 'location': 6, 'port': '2C'}, 
                {'numberToken': '8', 'resource': 'O', 'location': 7, 'port': ''}, 
                {'numberToken': '9', 'resource': 'O', 'location': 8, 'port': ''}, 
                {'numberToken': '6', 'resource': 'L', 'location': 9, 'port': ''}, 
                {'numberToken': '10', 'resource': 'C', 'location': 10, 'port': ''}, 
                {'numberToken': '2', 'resource': 'O', 'location': 11, 'port': ''}, 
                {'numberToken': '4', 'resource': 'C', 'location': 12, 'port': ''}, 
                {'numberToken': '5', 'resource': 'W', 'location': 13, 'port': ''}, 
                {'numberToken': '0', 'resource': 'D', 'location': 14, 'port': ''}, 
                {'numberToken': '3', 'resource': 'S', 'location': 15, 'port': '2L'}, 
                {'numberToken': '6', 'resource': 'C', 'location': 16, 'port': '2O'}, 
                {'numberToken': '12', 'resource': 'L', 'location': 17, 'port': ''}, 
                {'numberToken': '8', 'resource': 'W', 'location': 18, 'port': ''}
                ]
            }


threeTileSpots = [
      [0, 1, 4], [2, 5, 6], [1, 2, 5], [3, 4, 8], [0, 3, 4], [4, 5, 9],
      [1, 4, 5], [5, 6, 10], [3, 7, 8], [4, 8, 9], [5, 9, 10], [6, 10, 11],
      [7, 8, 12], [8, 9, 13], [9, 10, 14], [10, 11, 15], [8, 12, 13],
      [9, 13, 14], [10, 14, 15], [12, 13, 16], [13, 14, 17], [14, 15, 8],
      [13, 16, 17], [14, 17, 18]
    ]

twoTileSpots = [
      [0, 1], [1, 2], [0, 3], [2, 6], [6, 11], [11, 15], [15, 18],
      [18, 17], [16, 17], [12, 16], [7, 12], [3, 7]
    ]

# oneTileSpots = [0, 1, 2, 6, 11, 15, 18, 17, 16, 12, 7, 3]

best_first_settlements = []
best_second_settlements = []
all_first_settlements = []
all_second_settlements = []
production_wins = []
for index, row in dfSettlement.iterrows():
    if row['win'] == 1:
        production_wins.append(row['production'])
        # Concatenate values into strings for the first settlement
        first_settlement_str = f"{row['set1_num1']} {row['set1_res1']},{row['set1_num2']} {row['set1_res2']},{row['set1_num3']} {row['set1_res3']}"
        
        # Concatenate values into strings for the second settlement
        second_settlement_str = f"{row['set2_num1']} {row['set2_res1']},{row['set2_num2']} {row['set2_res2']},{row['set2_num3']} {row['set2_res3']}"
        

        # Append the strings to the lists
        best_first_settlements.append(first_settlement_str)
        best_second_settlements.append(second_settlement_str)

    first_settlement_str = f"{row['set1_num1']} {row['set1_res1']},{row['set1_num2']} {row['set1_res2']},{row['set1_num3']} {row['set1_res3']}"    
    second_settlement_str = f"{row['set2_num1']} {row['set2_res1']},{row['set2_num2']} {row['set2_res2']},{row['set2_num3']} {row['set2_res3']}"
    
    all_second_settlements.append(first_settlement_str)
    all_first_settlements.append(second_settlement_str)

    # Drop the specified columns
    dfCatan.drop(columns=['set1_num1', 'set1_res1', 'set1_num2', 'set1_res2', 'set1_num3', 'set1_res3',
                                'set2_num1', 'set2_res1', 'set2_num2', 'set2_res2', 'set2_num3', 'set2_res3'],
                      inplace=True, errors='ignore')

    # Add settlement strings to new columns
    dfCatan.loc[index, 'settlement1'] = first_settlement_str
    dfCatan.loc[index, 'settlement2'] = second_settlement_str


possible_settlements = []

# Define possible settlements for threeTileSpots
for spot_group in threeTileSpots:
    settlement_info = []
    for spot_index in spot_group:
        # Extract resource and numberToken information for each tile
        tile_info = f"{userBoard['tiles'][spot_index]['numberToken']} {userBoard['tiles'][spot_index]['resource']}"
        settlement_info.append(tile_info)

    # Append to possible_settlements as a single string
    possible_settlements.append(",".join(settlement_info))

# Define possible settlements for twoTileSpots
for spot_group in twoTileSpots:
    settlement_info = []
    for spot_index in spot_group:
        # Extract resource and numberToken information for each tile
        tile_info = f"{userBoard['tiles'][spot_index]['numberToken']} {userBoard['tiles'][spot_index]['resource']}"
        settlement_info.append(tile_info)
        # Append port information if available, and set numberToken to 0
        port = userBoard['tiles'][spot_index]['port']
        if port:
            port = ''.join(('0 ', port))
            settlement_info.append(port)
            

    # Append to possible_settlements as a single string
    possible_settlements.append(",".join(settlement_info))

print("Possible Settlements on User Board: ", possible_settlements)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Create a DataFrame
dfTrain = dfCatan[['gameNum', 'player', 'points', 'win', 'settlement1', 'settlement2', 'production']]
print(dfTrain.head())
# Split the data into training and testing sets
train_data, test_data = train_test_split(dfTrain, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and modeling
numeric_features = ['production']  # Assuming 'production' is the only numeric feature
numeric_transformer = ('num', StandardScaler(), numeric_features)
categorical_features = ['settlement1', 'settlement2']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        numeric_transformer,
        ('cat', categorical_transformer, categorical_features)])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Train the model
pipeline.fit(train_data.drop('win', axis=1), train_data['win'])

# Make predictions on the test set
predictions = pipeline.predict(test_data.drop('win', axis=1))

# Evaluate the model
accuracy = accuracy_score(test_data['win'], predictions)
print(f'Model Accuracy: {accuracy}')
# Now you can use the trained model to predict the winning settlements
new_settlements = possible_settlements  # Replace this with your new settlement data

# Transform the new settlement data using the preprocessor
new_settlements_transformed = preprocessor.transform(pd.DataFrame({'settlement1': new_settlements, 'settlement2': new_settlements}))

# Make predictions
predictions_new_settlements = pipeline.predict(new_settlements_transformed)

# Recommend settlements with a predicted win
recommended_settlements = [settlement for settlement, prediction in zip(new_settlements, predictions_new_settlements) if prediction == 1]
print('Recommended Settlements:', recommended_settlements)


