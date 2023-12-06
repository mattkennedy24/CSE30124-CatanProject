import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Load data from catanStats.csv file
# df = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')

# # Load data from catan_scores.csv file
# df_dataSet2 = pd.read_csv('../catan-helper/src/assets/data/catan_scores.csv')

# print("OG Col Names: ", df.columns)

# # Encode categorical columns
# label_encoder = LabelEncoder()
# for column in df.select_dtypes(include=['object']).columns:
#     df[column] = label_encoder.fit_transform(df[column])

# # Select rows where 'points' is greater than 10
# df_filtered = df[df['points'] > 10]

# # Calculate correlations with 'points' column
# correlations = df_filtered.corrwith(df_filtered['points'])

# # Display correlations
# print("Correlations with 'points' column:")
# print(correlations)

# # Find the column with the highest correlation
# highest_corr_column = correlations.abs().idxmax()
# print(f"The column with the highest correlation to 'points' is: {highest_corr_column}")
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

dfCatan = pd.read_csv('../catan-helper/src/assets/data/catanstats.csv')
print("OG Cols: ", dfCatan.columns)
dfCatan = dfCatan.rename(columns={'settlement1':'set1_num1', 'Unnamed: 16':'set1_res1', 'Unnamed: 17':'set1_num2', 'Unnamed: 18':'set1_res2', 'Unnamed: 19':'set1_num3', 'Unnamed: 20':'set1_res3', 'settlement2':'set2_num1', 'Unnamed: 22':'set2_res1', 'Unnamed: 23':'set2_num2', 'Unnamed: 24':'set2_res2', 'Unnamed: 25':'set2_num3', 'Unnamed: 26':'set2_res3'})
print("NEW Cols: ", dfCatan.columns)


dfCatan['win'] = (dfCatan['points'] >= 10).astype(int)
dfCatan['me'].fillna(0, inplace = True)
dfCatan.head()

#Is the position of the player important for winning?
fig = plt.figure(figsize=(13,4))

ax = fig.add_subplot(1,2,1)
player_win = dfCatan[dfCatan['win'] == 1]['player'].value_counts()
player_loss = dfCatan[dfCatan['win'] == 0]['player'].value_counts()/3

dfTempPlot = pd.DataFrame([player_win,player_loss])
dfTempPlot.index = ['Win','Loss']
dfTempPlot.plot(kind = 'bar',stacked = True, title = 'Winning depends on player position...', ax=ax)
ax.set_ylabel('Games (scaled "loss" bar for comparison)')

#Does this also hold for "me"?
ax2 = fig.add_subplot(1,2,2)
me_win = dfCatan[(dfCatan['win'] == 1) & (dfCatan['me'] == 1.0)]['player'].value_counts()
me_loss = dfCatan[(dfCatan['win'] == 0) & (dfCatan['me'] == 1.0)]['player'].value_counts()

dfTempPlot = pd.DataFrame([me_win,me_loss])
dfTempPlot.index = ['Win','Loss']
dfTempPlot.plot(kind = 'bar',stacked = True, title = '... also for "me" - being player 2 is a good thing', ax=ax2)
ax2.set_ylabel('Games')

print('"me" wins ' + str(100*sum(dfCatan[dfCatan['me'] == 1.0]['win'])/float(max(dfCatan['gameNum']))) + '% of all games!')

#Let's look at some correlations - small data set, so take p-value with grain of salt
from scipy.stats import pearsonr

fig = plt.figure(figsize=(15,6))
ax = fig.add_subplot(2,3,1)
ax.scatter(dfCatan['production'], dfCatan['points'], c='black')
ax.set_title('Points vs Production')
ax.text(20, 12, 'r = '+ str(round(pearsonr(dfCatan['production'], dfCatan['points'])[0],2)))

ax2 = fig.add_subplot(2,3,2)
ax2.scatter(dfCatan['tradeGain'], dfCatan['points'], c='black')
ax2.set_title('Points vs Trade gain')
ax2.text(0, 12, 'r = '+ str(round(pearsonr(dfCatan['tradeGain'], dfCatan['points'])[0],2)))

ax3 = fig.add_subplot(2,3,3)
ax3.scatter(dfCatan['robberCardsGain'], dfCatan['points'], c='black')
ax3.set_title('Points vs Robber cards gain')
ax3.text(0, 12, 'r = '+ str(round(pearsonr(dfCatan['robberCardsGain'], dfCatan['points'])[0],2)))

ax4 = fig.add_subplot(2,3,4)
ax4.scatter(dfCatan['tribute'], dfCatan['points'], c='red')
ax4.set_title('Points vs Tribute')
ax4.text(0, 1, 'r = '+ str(round(pearsonr(dfCatan['tribute'], dfCatan['points'])[0],2)))

ax5 = fig.add_subplot(2,3,5)
ax5.scatter(dfCatan['tradeLoss'], dfCatan['points'], c='red')
ax5.set_title('Points vs Trade loss')
ax5.text(0, 1, 'r = '+ str(round(pearsonr(dfCatan['tradeLoss'], dfCatan['points'])[0],2)))

ax6 = fig.add_subplot(2,3,6)
ax6.scatter(dfCatan['robberCardsLoss'], dfCatan['points'], c='red')
ax6.set_title('Points vs Robber cards loss')
ax6.text(0, 1, 'r = '+ str(round(pearsonr(dfCatan['robberCardsLoss'], dfCatan['points'])[0],2)))

fig.tight_layout()
plt.show()