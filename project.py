#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:51:16 2021

@author: baber
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import seaborn as sns



#import data
data = pd.read_csv('/Users/baber/Library/CloudStorage/OneDrive-City,UniversityofLondon/Documents/ML Project/fourteendec/bank-additional-full.csv', sep = ';')
data = data.drop('duration', axis = 1)
data= data.drop('default', axis = 1) 
data['pdays'] = data['pdays'].replace({-1: 0})
bank=data

#change 999 in pdays to 0
#replace unknown with mode
#bank = bank.replace('unknown', np.NaN)
#bank = bank.fillna(bank.mode().iloc[0])
bank.pdays.replace({999: 0}, inplace=True)
#bank.dropna(inplace=True)

#cateroical variabes
bank[['job', 'marital', 'education', 'contact', 'month', 'poutcome']].apply(lambda x: x.astype('category'))

#Convert two factor categorical to binary - 'y' Responce variable
yesno = {'yes': 1,'no': 0} # for default, housing, loan, y
# cellularyes = {'cellular': 1, 'telephone': 0}
# bank.housing = [yesno[item] for item in bank.housing]
# bank.loan = [yesno[item] for item in bank.loan]
bank.y = [yesno[item] for item in bank.y]
# bank.contact = [cellularyes[item] for item in bank.contact]
#one hot non binary predictors
# one_hot = pd.get_dummies(X, columns = ['job','marital','education','contact', 'month','day_of_week','poutcome'])
# bank = one_hot


#Stratified Split: 70% training, 30% test
train, test = train_test_split(bank, test_size=0.3, stratify = bank.y, random_state = 98)


train.to_csv('/Users/baber/Library/CloudStorage/OneDrive-City,UniversityofLondon/Documents/ML Project/fourteendec/train.csv', index = False)
test.to_csv('/Users/baber/Library/CloudStorage/OneDrive-City,UniversityofLondon/Documents/ML Project/fourteendec/test.csv', index = False)


#visualizations
numerical = [
  'age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
  ]

categorical = [
  'job','marital','education', 'housing', 'loan', 'contact', 'month',
  'day_of_week','campaign', 'pdays','previous', 'poutcome',
]

sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})


bank[numerical].hist(bins=15, figsize=(15, 6), layout=(2, 4));

fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(bank[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
        
sns.countplot(bank['y']);

bankn = bank[[numerical]]

classes = data['y'].values

plt.matshow(bank.corr(method = 'pearson'))

cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)



