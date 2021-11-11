import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import keras

dataset = pd.read_csv('/Users/jasonfang/Work/Dataset/Practical/creditcard.csv')

dataset.head()
print(dataset.columns)
dataset.describe()
#find out na columns
dataset.isna().any()
dataset.isna().sum()


# fig = plt.figure(figsize=(15, 8))
# plt.suptitle('Histograms of Numerical Columns', fontsize=20)
# for i in range(dataset.shape[1]):
#     plt.subplot(6, 3, i + 1)
#     f = plt.gca()
#     f.set_title(dataset.columns.values[i])
#
#     vals = np.size(dataset.iloc[:, i].unique())
#     if vals >= 100:
#         vals = 100
#
#     plt.hist(dataset.iloc[:, i], bins=vals, color='#3F5D7D')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

from sklearn.preprocessing import StandardScaler

dataset['normalizedAmount'] = StandardScaler().fit_transform(dataset['Amount'].values.reshape(-1,1))

data = dataset.drop(['Amount','Time'],axis=1)


X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

#0.0017
y.mean()
#284807
len(dataset)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

X_test.shape
#shape 29

model = Sequential([
    Dense(units=16, input_dim = 29,activation='relu'),
    Dense(units=24,activation='relu'),
    Dropout(0.5),
    Dense(20,activation='relu'),
    Dense(24,activation='relu'),
    Dense(1,activation='sigmoid'),
])

model.summary()
#training
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=15,epochs=5)

score = model.evaluate(X_test, y_test)

print(score)

import matplotlib.pyplot as plt
import itertools

from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    cnf_matrix = confusion_matrix(y_test, y_pred.round())

    print(cnf_matrix)

    plot_confusion_matrix(cnf_matrix, classes=[0, 1])
    plt.show()

    df_cm = pd.DataFrame(cnf_matrix, index=(0, 1), columns=(0, 1))
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g')


fraud_indices = np.array(data[data.Class == 1].index)
number_records_fraud = len(fraud_indices)
print(number_records_fraud)

normal_indices = data[data.Class == 0].index
#randomly select
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)
print(len(random_normal_indices))

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
print(len(under_sample_indices))

under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.iloc[:,under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:,under_sample_data.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample, test_size=0.3)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)

y_pred = model.predict(X_test)
y_expected = pd.DataFrame(y_test)
cnf_matrix = confusion_matrix(y_expected, y_pred.round())
plot_confusion_matrix(cnf_matrix, classes=[0,1])
plt.show()

df_cm = pd.DataFrame(cnf_matrix, index=(0, 1), columns=(0, 1))
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')

#smote oversampleing
from imblearn.over_sampling import SMOTE

X_resample, y_resample = SMOTE().fit_resample(X,y.values.ravel())
y_resample = pd.DataFrame(y_resample)
X_resample = pd.DataFrame(X_resample)


X_train, X_test, y_train, y_test = train_test_split(X_resample,y_resample,test_size=0.3)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=15,epochs=5)
