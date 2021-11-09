#### Importing Libraries ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = pd.read_csv('P39-Minimizing-Churn-Data/churn_data.csv')
# Users who were 60 days enrolled, churn in the next 30

dataset.head()
print(dataset.columns)
dataset.describe()

#find out na columns
dataset.isna().any()
dataset.isna().sum()
dataset = dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])

