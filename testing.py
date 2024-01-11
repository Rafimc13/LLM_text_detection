import pandas as pd


# Read the train essays .csv
essays_df = pd.read_csv('data/best_train_essays.csv')
essays_cols = essays_df.columns.tolist()

for i in range(len(essays_df)):
    j=0
    if essays_df.loc[i, essays_cols[3]] == 1:
        essays_df.loc[i, 'cluster'] = 'LLM_C' +str(LLM[j])
        j+=1