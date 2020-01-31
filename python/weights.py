# FULL null

def give_weight(n12,label):
    column = 1 if label == 'COOP' else 0
    row = 10 if n12 >= 10 else n12
    #print(n12,label,np.floor(weights[row,column]/min(weights[:,column])))
    return np.floor(weights[row,column]/min(weights[:,column]))

weights = np.zeros((11,2))
lengths = np.array([len(features[(features['N_12'] == 0)]),
           len(features[(features['N_12'] == 1)]),
           len(features[(features['N_12'] == 2)]),
           len(features[(features['N_12'] == 3)]),
           len(features[(features['N_12'] == 4)]),
           len(features[(features['N_12'] == 5)]),
           len(features[(features['N_12'] == 6)]),
           len(features[(features['N_12'] == 7)]),
           len(features[(features['N_12'] == 8)]),
           len(features[(features['N_12'] == 9)]),
           len(features[(features['N_12'] >= 10)]),
          ]
                  )
print(lengths)
weights[:,0] = lengths/min(lengths)
features['WEIGHTS'] = features.apply(lambda row:give_weight(row[2],row[5]),axis=1)
tr_features = pd.merge(tr_labels,features,left_on=['TF1','TF2'],right_on=['P1','P2'])

lengths = np.array([len(tr_features[(tr_features['N_12'] == 0) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 1) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 2) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 3) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 4) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 5) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 6) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 7) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 8) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] == 9) & (tr_features['LABEL'] == 'COOP')]),
           len(tr_features[(tr_features['N_12'] >= 10) & (tr_features['LABEL'] == 'COOP')]),
          ]
                  )
weights[:,1] = lengths/min(lengths)
print(weights)
weights = pd.DataFrame(weights,index=range(0,11),columns=['NINT','COOP'])
print(weights)
with open('nautica_weights.csv','w') as ofile:
    weights.to_csv(ofile,sep=',')

#features['WEIGHTS'] = features.apply(lambda row:give_weight(row[2],row[5]),axis=1)
ts_features = pd.merge(ts_labels,features,left_on=['TF1','TF2'],right_on=['P1','P2']).drop(['P1','P2'],axis=1)
ts_features['WEIGHTS'] = ts_features[['N_12','LABEL']].apply(lambda row:give_weight(row[0],row[1]),axis=1)
print(ts_features)


####



