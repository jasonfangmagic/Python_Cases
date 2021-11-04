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

sns.countplot(df_cancer['target'])

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)

plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(), annot= True)

#feature scaling

#create the model

X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#normalize data
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train - min_train)/range_train
sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)
sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)

min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test_scaled = (X_test - min_test)/range_test

#standardize

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

sns.scatterplot(x = X_train_s['mean area'], y = X_train_s['mean smoothness'], hue = y_train)




from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

classifier = SVC(kernel = 'rbf', random_state = 0)

classifier.fit(X_train_s,y_train)

y_pred = classifier.predict(X_test_s)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot = True)


from sklearn.metrics import f1_score
f1 = f1_score(y_test, y_pred, average='macro')
print(f1)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)
print(acc)

#grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1, 10, 100], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train_scaled, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)



