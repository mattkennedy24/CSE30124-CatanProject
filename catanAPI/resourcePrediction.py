import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np



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

resource_mapping = {'L': 0, 'C': 1, 'S': 2, 'W': 3, 'O': 4, 'D': 5, 'B': 5}
port_mapping = {'': 0,'2L': 6, '2C': 7, '2S': 8, '2W': 9, '2O': 10, '3G': 11, 'B': 5}

for tile in userBoard['tiles']:
    tile['numberToken'] = int(tile['numberToken'])
    tile['resource'] = resource_mapping.get(tile['resource'], tile['resource'])
    tile['port'] = port_mapping.get(tile['port'], tile['port'])


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

dfCatanStats = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

# Create a new column called win that is equal to one if the row's points value >= 10
dfCatanStats['win'] = (dfCatanStats['points'] >= 10).astype(int)
dfCatanStats = dfCatanStats.drop(columns=['me'], axis=1)

# rename settlement position column headers
dfCatanStats = dfCatanStats.rename(columns={'settlement1':'set1a', 'Unnamed: 17':'set1b', 'Unnamed: 19':'set1c', 'settlement2':'set2a', 'Unnamed: 23':'set2b', 'Unnamed: 25':'set2c'})

# convert 'Unnamed: ##' labels to a numerical variable in new terms (columns)
def resources (new, old):
    dfCatanStats[new] = dfCatanStats[old].map({'L':0, 'C':1, 'S':2, 'W':3, 'O':4, 'D': 5, '2L':6, '2C':7, '2S':8, '2W':9, '2O':10, '3G':11, 'B':5})

newre = ['res1a', 'res1b', 'res1c', 'res2a', 'res2b', 'res2c']
oldre = ['Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 26']

for new, old in zip(newre, oldre):
    resources (new, old)
    
# dropping unused columns
dfCatanStats = dfCatanStats.drop(['Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 26'], axis=1)

# convert the 'settlement' labels into numerical terms (columns)
def numchance (new, old):
    dfCatanStats[new] = dfCatanStats[old].map({0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1})

newset = ['nc1a', 'nc1b', 'nc1c', 'nc2a', 'nc2b', 'nc2c']
oldset = ['set1a', 'set1b', 'set1c', 'set2a', 'set2b', 'set2c']

for new, old in zip(newset, oldset):
    numchance(new, old)

dfCatanStats['totalChance_1'] = dfCatanStats[:][['nc1a', 'nc1b', 'nc1c']].sum(axis=1)
dfCatanStats['totalChance_2'] = dfCatanStats[:][['nc2a', 'nc2b', 'nc2c']].sum(axis=1)

dfCatanStats = dfCatanStats.drop(columns=newset)

# Create an empty DataFrame with the desired column names
columns = ['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3']
dfUserBoard = pd.DataFrame(columns=columns)

# Iterate through threeTileSpots and populate the DataFrame
for spot in threeTileSpots:
    row_data = []
    for location in spot:
        tile = userBoard['tiles'][location]
        row_data.extend([tile['numberToken'], tile['resource']])
    dfUserBoard = pd.concat([dfUserBoard, pd.DataFrame([row_data], columns=columns)], ignore_index=True)

# Iterate through twoTileSpots and populate the DataFrame
for spot in twoTileSpots:
    row_data = []
    for location in spot:
        tile = userBoard['tiles'][location]
        row_data.extend([tile['numberToken'], tile['resource']])

    # Check if there is a third entity (port) in the spot
    if userBoard['tiles'][spot[-1]]['port'] > 0:
        port_location = spot[-1]
        port_tile = userBoard['tiles'][port_location]
        row_data.extend([0, port_tile['port']])
    else:
        row_data.extend([0, 0])  # Set number3 to 0 and resource3 to 0
    
    dfUserBoard = pd.concat([dfUserBoard, pd.DataFrame([row_data], columns=columns)], ignore_index=True)


# Function to map 'numberX' columns and calculate sum
def calculate_num_chance(row):
    num_columns = ['number1', 'number2', 'number3']
    mapped_values = row[num_columns].map({0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1})
    return mapped_values.sum()

# Apply the function to create 'numChance' column
dfUserBoard['totalChance'] = dfUserBoard.apply(lambda row: calculate_num_chance(row), axis=1)

# Ensure 'totalChance' column is of integer data type
dfUserBoard['totalChance'] = dfUserBoard['totalChance'].astype(int)

# Display the resulting DataFrame
# print(dfUserBoard)

