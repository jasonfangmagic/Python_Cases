# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()

print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])
print(cancer['feature_names'])
print(cancer['data'].shape)

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))

df_cancer.head()
df_cancer.tail()

#visualizing the data


sns.pairplot(df_cancer, hue = 'target',vars = ['mean radius', 'mean texture','mean perimeter', 'mean area',
 'mean smoothness'], height=2)







