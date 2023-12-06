import copy
from flask import Flask, Response, jsonify, request, send_file
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns

from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*',  supports_credentials=True)

# Load data from catanStats.csv file
df_dataSet1 = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

# Load data from catan_scores.csv file
df_dataSet2 = pd.read_csv('../catan-helper/src/assets/data/catan_scores.csv')

@app.route('/api/load_dataSet1', methods=['GET'])
def load_dataSet1():
    # Convert data to JSON and return
    data_json = df_dataSet1.to_json(orient='records')
    return jsonify(data_json)

@app.route('/api/load_dataSet2', methods=['GET'])
def load_dataSet2():
    # Convert data to JSON and return
    data_json = df_dataSet2.to_json(orient='records')
    return jsonify(data_json)

@app.route('/api/recommend_settlement', methods=['POST'])
def recommend_settlement():

    # Load data from catanStats.csv file
    dfCatanStats = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')
    # Get the user's board data
    userBoard = request.json.get('plainBoardData') 
    # Create a deep copy of the userBoard
    userBoardOG = copy.deepcopy(userBoard) 

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

    # Convert User Board's Categorical Resource and Port Data to Numeric for Analysis
    resource_mapping = {'L': 0, 'C': 1, 'S': 2, 'W': 3, 'O': 4, 'D': 5}
    port_mapping = {'': 12,'2L': 6, '2C': 7, '2S': 8, '2W': 9, '2O': 10, '3G': 11}

    for tile in userBoard['tiles']:
        tile['numberToken'] = int(tile['numberToken'])
        tile['resource'] = resource_mapping.get(tile['resource'], tile['resource'])
        tile['port'] = port_mapping.get(tile['port'], tile['port'])

    # Add a new column to dataframe called win that is equal to one if the row's points value >= 10 (aka they won)
    dfCatanStats['win'] = (dfCatanStats['points'] >= 10).astype(int)
    dfCatanStats = dfCatanStats.drop(columns=['me'], axis=1)

    # rename settlement number token column headers for clarity
    dfCatanStats = dfCatanStats.rename(columns={'settlement1':'set1a', 'Unnamed: 17':'set1b', 'Unnamed: 19':'set1c', 'settlement2':'set2a', 'Unnamed: 23':'set2b', 'Unnamed: 25':'set2c'})

    # Convert the resource data from categorical to numeric for the data set too
    def resources (new, old):
        dfCatanStats[new] = dfCatanStats[old].map({'L':0, 'C':1, 'S':2, 'W':3, 'O':4, 'D': 5, '2L':6, '2C':7, '2S':8, '2W':9, '2O':10, '3G':11, 'B':5})

    newCols = ['res1a', 'res1b', 'res1c', 'res2a', 'res2b', 'res2c']
    oldCols = ['Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 26']

    for new, old in zip(newCols, oldCols):
        resources (new, old)
    
    # drop unused columns
    dfCatanStats = dfCatanStats.drop(['Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22', 'Unnamed: 24', 'Unnamed: 26'], axis=1)

    # to calculate the totalChance, aka the sum of the dots on the number token cards, convert the number to the corresponding number of dots
    def numchance (new, old):
        dfCatanStats[new] = dfCatanStats[old].map({0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1})

    newSet = ['nc1a', 'nc1b', 'nc1c', 'nc2a', 'nc2b', 'nc2c']
    oldSet = ['set1a', 'set1b', 'set1c', 'set2a', 'set2b', 'set2c']

    for new, old in zip(newSet, oldSet):
        numchance(new, old)

    # create two new columns to hold the totalChance for each first and second settlement placed by each player in each game
    dfCatanStats['totalChance_1'] = dfCatanStats[:][['nc1a', 'nc1b', 'nc1c']].sum(axis=1)
    dfCatanStats['totalChance_2'] = dfCatanStats[:][['nc2a', 'nc2b', 'nc2c']].sum(axis=1)

    # drop the dots columns because they are no longer needed
    dfCatanStats = dfCatanStats.drop(columns=newSet)

    # Create a new empty DataFrame with the desired column names for the user board data
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

        # Check if there is a port in the spot for only the twoTileSpots 
        if userBoard['tiles'][spot[-1]]['port'] > 0:
            port_location = spot[-1]
            port_tile = userBoard['tiles'][port_location]
            row_data.extend([0, port_tile['port']])
        else:
            row_data.extend([0, 0])  # Set number3 to 0 and resource3 to 0
    
        dfUserBoard = pd.concat([dfUserBoard, pd.DataFrame([row_data], columns=columns)], ignore_index=True)


    # Function to map 'number[X]' columns and calculate sum (doing the same conversion from number to dots representation to get totalChance)
    def calculate_num_chance(row):
        num_columns = ['number1', 'number2', 'number3']
        mapped_values = row[num_columns].map({0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1})
        return mapped_values.sum()

    # Apply the function to create 'totalChance' column for user board data
    dfUserBoard['totalChance'] = dfUserBoard.apply(lambda row: calculate_num_chance(row), axis=1)

    # Ensure 'totalChance' column is of integer data type
    dfUserBoard['totalChance'] = dfUserBoard['totalChance'].astype(int)

    #################################
    ###### Logistic Regression ######
    #################################

    # Prepare the training data on the first settlement statistics
    # the data frames for the user board and the catanStats must have the same exact structure for algorithms to process
    X_first_settlement = dfCatanStats[['set1a','res1a', 'set1b', 'res1b', 'set1c', 'res1c', 'totalChance_1']]
    X_first_settlement = X_first_settlement.rename(columns={
        'set1a':'number1', 'res1a':'resource1', 'set1b':'number2', 
        'res1b':'resource2', 'set1c':'number3', 'res1c':'resource3', 
        'totalChance_1':'totalChance'})

    X_second_settlement =  dfCatanStats[['set2a','res2a', 'set2b', 'res2b', 'set2c', 'res2c', 'totalChance_2']]
    X_second_settlement = X_second_settlement.rename(columns={
        'set2a':'number1', 'res2a':'resource1', 'set2b':'number2', 
        'res2b':'resource2', 'set2c':'number3', 'res2c':'resource3', 
        'totalChance_2':'totalChance'})

    y = dfCatanStats['win']

    # Split the data into training and testing sets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_first_settlement, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_second_settlement, y, test_size=0.2, random_state=42)
    
    # Create and train a logistic regression model for both first and second settlements
    model1= LogisticRegression()
    model2 = LogisticRegression()
    model1.fit(X_train1, y_train1)
    model2.fit(X_train2, y_train2)

    # Evaluate the model on the test sets
    y_pred1 = model1.predict(X_test1)
    y_pred2 = model2.predict(X_test2)
    accuracy = (accuracy_score(y_test1, y_pred1) + accuracy_score(y_test2, y_pred2)) / 2

    # Now, use the trained model to predict 'win' for dfUserBoard
    user_predictions1 = model1.predict(dfUserBoard)
    user_predictions2 = model2.predict(dfUserBoard)

    # Add the predictions to dfUserBoard
    dfUserBoard['pw_logref_1'] = user_predictions1
    dfUserBoard['pw_logreg_2'] = user_predictions2

    #######################################
    ###### Decision Treer Classifier ######
    #######################################

    # Create and train a Decision Tree Classifier for the first settlement
    dt_model_first_settlement = DecisionTreeClassifier(random_state=42)
    dt_model_first_settlement.fit(X_first_settlement, y)

    
    # Create and train a Decision Tree Classifier for the second settlement
    dt_model_second_settlement = DecisionTreeClassifier(random_state=42)
    dt_model_second_settlement.fit(X_second_settlement, y)

    # Now, use the trained models to predict 'win' for dfUserBoard
    user_board_predictions_first_settlement = dt_model_first_settlement.predict(dfUserBoard[['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3', 'totalChance']])
    user_board_predictions_second_settlement = dt_model_second_settlement.predict(dfUserBoard[['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3', 'totalChance']])

    # Add the predictions to dfUserBoard
    dfUserBoard['pw_dt_1'] = user_board_predictions_first_settlement
    dfUserBoard['pw_dt_2'] = user_board_predictions_second_settlement

    ##################################################
    ###### Principal Component Analysis with DT ######
    ##################################################


    # Apply PCA
    pca = PCA(n_components=3)  # this number yielded the best/most accurate results
    X_first_settlement_pca = pca.fit_transform(X_first_settlement)
    X_second_settlement_pca = pca.fit_transform(X_second_settlement)

    # Concatenate the principal components with other features
    X_first_settlement_combined = pd.concat([X_first_settlement, pd.DataFrame(X_first_settlement_pca, columns=['PC1', 'PC2', 'PC3'])], axis=1)
    X_second_settlement_combined = pd.concat([X_second_settlement, pd.DataFrame(X_second_settlement_pca, columns=['PC1', 'PC2', 'PC3'])], axis=1)

    # Create and train a Decision Tree Classifier for the first settlement with PCA
    dt_model_first_settlement_pca = DecisionTreeClassifier(random_state=42)
    dt_model_first_settlement_pca.fit(X_first_settlement_combined, dfCatanStats['win'])

    # Create and train a Decision Tree Classifier for the second settlement with PCA
    dt_model_second_settlement_pca = DecisionTreeClassifier(random_state=42)
    dt_model_second_settlement_pca.fit(X_second_settlement_combined, dfCatanStats['win'])

    # Now, use the trained models to predict 'win' for dfUserBoard with PCA
    user_board_pca_first_settlement = pca.transform(dfUserBoard[['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3', 'totalChance']])
    dfUserBoard_pca_first_settlement = pd.concat([dfUserBoard, pd.DataFrame(user_board_pca_first_settlement, columns=['PC1', 'PC2', 'PC3'])], axis=1)
    user_board_predictions_first_settlement_pca = dt_model_first_settlement_pca.predict(dfUserBoard_pca_first_settlement[['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3', 'totalChance', 'PC1', 'PC2', 'PC3']])

    user_board_pca_second_settlement = pca.transform(dfUserBoard[['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3', 'totalChance']])
    dfUserBoard_pca_second_settlement = pd.concat([dfUserBoard, pd.DataFrame(user_board_pca_second_settlement, columns=['PC1', 'PC2', 'PC3'])], axis=1)
    user_board_predictions_second_settlement_pca = dt_model_second_settlement_pca.predict(dfUserBoard_pca_second_settlement[['number1', 'resource1', 'number2', 'resource2', 'number3', 'resource3', 'totalChance', 'PC1', 'PC2', 'PC3']])

    # Add the predictions to dfUserBoard
    dfUserBoard['pw_dt_pca_1'] = user_board_predictions_first_settlement_pca
    dfUserBoard['pw_dt_pca_2'] = user_board_predictions_second_settlement_pca

    # Reverse resource_mapping and port_mapping dictionaries
    rev_map = {'L': 0, 'C': 1, 'S': 2, 'W': 3, 'O': 4, 'D': 5, '': 12,'2L': 6, '2C': 7, '2S': 8, '2W': 9, '2O': 10, '3G': 11}
    reverse_resource_mapping = {v: k for k, v in rev_map.items()}

    # Convert numerical values back to original string values in dfUserBoard
    dfUserBoard[['resource1', 'resource2', 'resource3']] = dfUserBoard[['resource1', 'resource2', 'resource3']].map(lambda x: reverse_resource_mapping.get(x, ''))

    # Specify the columns to check
    first_settlement_cols = ['pw_logref_1', 
                        'pw_dt_1', 
                        'pw_dt_pca_1']
    second_settlement_cols = ['pw_logreg_2', 
                        'pw_dt_2', 
                        'pw_dt_pca_2']


    # Use boolean indexing to filter rows with at least one 1 in the specified columns
    first_recs = dfUserBoard[dfUserBoard[first_settlement_cols].eq(1).any(axis=1)]
    second_recs = dfUserBoard[dfUserBoard[second_settlement_cols].eq(1).any(axis=1)]

    # Get the index of the resulting rows
    first_settlement_result_indices = first_recs.index
    second_settlement_result_indices = second_recs.index

    # Combined list of settlement locations
    combined_settlements = threeTileSpots + twoTileSpots

    # Initialize dictionaries to store recommendations
    firstSettlementRecommendations = {}
    secondSettlementRecommendations = {}

    # Loop through first settlement indices
    for i, index in enumerate(first_settlement_result_indices, start=1):
        settlement_locations = combined_settlements[index]
        recommendations = [userBoardOG['tiles'][location] for location in settlement_locations]
        firstSettlementRecommendations[i] = recommendations

    # Loop through second settlement indices
    for i, index in enumerate(second_settlement_result_indices, start=1):
        settlement_locations = combined_settlements[index]
        recommendations = [userBoardOG['tiles'][location] for location in settlement_locations]
        secondSettlementRecommendations[i] = recommendations

    recommendations = [firstSettlementRecommendations, secondSettlementRecommendations]

    # Return the recommendation (replace with your actual recommendation)
    return jsonify({'recommendation': recommendations})


@app.route('/api/position_importance_chart', methods=['GET'])
def position_importance_chart():
    # Load data from catanStats.csv file
    dfCatanStats = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')
    dfCatanStats['win'] = (dfCatanStats['points'] >= 10).astype(int)
    # Prepare data
    player_win_counts = dfCatanStats[dfCatanStats['win'] == 1]['player'].value_counts()
    player_loss_counts = dfCatanStats[dfCatanStats['win'] == 0]['player'].value_counts() / 3

    df_temp_plot = pd.DataFrame([player_win_counts, player_loss_counts])
    df_temp_plot.index = ['Win', 'Loss']

    # Create a stacked bar chart
    fig, ax = plt.subplots(figsize=(13, 4))
    df_temp_plot.plot(kind='bar', stacked=True, title='Winning depends on player position...', ax=ax)
    ax.set_ylabel('Games (scaled "loss" bar for comparison)')

    # Save the plot to BytesIO
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Return the image as a response
    return send_file(image_stream, mimetype='image/png', as_attachment=True, download_name='position_importance_chart.png')


@app.route('/api/dice_roll_plot', methods=['GET'])
def dice_roll_plot():
    dfCatanStats = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

    dice_data = pd.DataFrame(dfCatanStats.iloc[:, list(range(4, 15))].sum(), columns=["totals"])
    dice_data["rolls"] = range(2, 13)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dice_data["rolls"], y=dice_data["totals"])
    plt.title('Dice Roll Totals')
    plt.xlabel('Rolls')
    plt.ylabel('Totals')

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)

    # Return the image as a response
    return send_file(image_stream, mimetype='image/png', as_attachment=True, download_name='dice_roll_plot.png')

@app.route('/api/feature_importance_plot', methods=['GET'])
def feature_importances_plot():
    dfCatanStats = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

    dfCatanStats['win'] = (dfCatanStats['points'] >= 10).astype(int)

        # Drop irrelevant columns for clustering
    dfCatanStats = dfCatanStats.drop(columns=['gameNum', 'points', 'player', 'me', 'settlement1', 'Unnamed: 16', 'Unnamed: 17',
        'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'settlement2',
        'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
        'Unnamed: 26'])
    
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

    # Save the plot to BytesIO
    img_bytesio = BytesIO()
    plt.savefig(img_bytesio, format='png')
    img_bytesio.seek(0)

    # Clear the plot for the next request
    plt.clf()

    # Return the image as a Flask response
    return Response(img_bytesio.getvalue(), content_type='image/png')