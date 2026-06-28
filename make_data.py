import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

### make df
def DF_dict(data, itv, par, iter):
    scaler = MinMaxScaler()
    part_arr = np.array([1]*par + [2]*(iter))
    temp_df = {}
    for i in range(len(data)-itv +1):
        temp = data.iloc[i:i+itv].copy()
        temp.reset_index(drop=True, inplace=True)
        temp['part'] = part_arr
        temp['N_total'] = scaler.fit_transform(temp[['total']])
        key = i
        temp_df[key] = temp
    return temp_df

def LR_dict(df_dict):
    temp_lr = {}
    for key in df_dict.keys():
        temp = df_dict[key]
        temp_lr[key] = {}
        for p in [1,2]:
            lr = LinearRegression()
            X_data = temp[temp.part == p].idx
            y_data = temp[temp.part == p].N_total
            
            lr.fit(X_data.values.reshape(-1,1), y_data)
            lr_pred = lr.predict(X_data.values.reshape(-1,1))
        
            temp_lr[key]['LR' + str(p)] = lr
            temp_lr[key]['LR' + str(p)+ 'val'] = lr_pred
    return temp_lr

### make df
def make_df(data, itv, par, iter): 
    dic1 = DF_dict(data, itv, par, iter)
    dic2 = LR_dict(dic1)
    cols = ['data_num','part1_patient_mean','part2_patient_mean',] + ['part1_std','part2_std'] + ['week','part1_mean','part1_slope','part2_mean','part2_slope']

    features = pd.DataFrame()
    for key in dic1.keys():
        data_tmp = dic1[key]
        add = [key, 
               data_tmp[data_tmp.part==1].total.mean(), 
               data_tmp[data_tmp.part==2].total.mean()]
        add.append(data_tmp[data_tmp.part==1].N_total.std())
        add.append(data_tmp[data_tmp.part==2].N_total.std())
        add.append(data_tmp.date.iloc[0].weekday())
        for j in [1,2]:
            temp = dic2[key]['LR'+str(j)]
            add += [dic2[key]['LR'+str(j)+'val'].mean(),temp.coef_[0]]
        features = pd.concat([features, pd.DataFrame(add).T])

    features.reset_index(drop=True, inplace=True)
    features.columns = cols
    features['mean_diff'] = abs(features.part2_mean - features.part1_mean)
    features['slope_diff'] = abs(features.part2_slope - features.part1_slope)
    features['slope_ratio'] = features.part2_slope / features.part1_slope    

    ### add N_total 
    temp = []
    for num in list(features.data_num):
        row = np.concatenate([
                dic1[int(num)][:par][['policy','Delta','Omicron']].mean().values,
                dic1[int(num)][par:][['policy','Delta','Omicron']].mean().values
                ])
        temp.append(row)
    temp = pd.DataFrame(temp)
    temp.columns = ['policy1','Delta1','Omicron1'] + ['policy2','Delta2','Omicron2']
    features = pd.concat([features,temp],axis=1)
    return features

def make_label(dataframe, index_name, cls, label_name):
    (a,b,c) = (1,0.01,0.01)
    dataframe[str(index_name)] = a * np.sinh(b*(dataframe.part2_patient_mean - dataframe.part1_patient_mean) / (dataframe.part1_patient_mean)) * np.exp(c * (dataframe.part2_slope - dataframe.part1_slope))
    dataframe = dataframe.sort_values(by=index_name)
    q = len(dataframe) // cls
    r = len(dataframe) % cls
    dataframe[str(label_name)] = np.concatenate([np.zeros(q) + c for c in range(cls)] + [np.zeros(r)+ cls-1])
    return dataframe

# input
n,m,t = (35,21,14)

### load data
case = pd.read_excel("data/data.xlsx")
case['idx'] = case.index

# make data
dict1 = DF_dict(case,n,m,t)
df = make_df(case,n,m,t)
df = make_label(df,'RI',3,'label')
df.reset_index(drop=True, inplace=True)

### save
df.to_csv('result/pre_data.csv')