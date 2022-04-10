import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import csv
from sklearn.model_selection import train_test_split

# visualization
import seaborn as sns
import matplotlib.pyplot as plt


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


path_train_input = r'data\train.csv'
path_test_input = r'data\test.csv'

pd.set_option('display.max_columns', None)
train_df = pd.read_csv(path_train_input)
test_df = pd.read_csv(path_test_input)
combine = [train_df, test_df]
columns = train_df.columns.values
#print(train_df.info())
# print(train_df.describe())
# print(train_df.describe(include=['O']))

# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Age', bins=20)
# plt.show()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
'''grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.show()'''
'''grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()'''
'''grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()'''

#print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
#print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#print(pd.crosstab(train_df['Title'], train_df['Sex']))
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

for dataset in combine:
    dataset['Mr'] = 0
    dataset['Mss'] = 0
    dataset['Mrs'] = 0
    dataset['Master'] = 0
    dataset['Rare'] = 0
    dataset.loc[dataset['Title'] == 'Mr', 'Mr'] = 1
    dataset.loc[dataset['Title'] == 'Mss', 'Mss'] = 1
    dataset.loc[dataset['Title'] == 'Mrs', 'Mrs'] = 1
    dataset.loc[dataset['Title'] == 'Master', 'Master'] = 1
    dataset.loc[dataset['Title'] == 'Rare', 'Rare'] = 1

train_df = train_df.drop(['Name', 'PassengerId', 'Title'], axis=1)
test_df = test_df.drop(['Name', 'Title'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

'''grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()'''

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)
'''
train_df['AgeBand'] = pd.cut(train_df['Age'], 6)

for dataset in combine:
    dataset['13.333'] = 0
    dataset['26.667'] = 0
    dataset['40.0'] = 0
    dataset['53.333'] = 0
    dataset['66.667'] = 0
    dataset['66.667+'] = 0
    dataset.loc[dataset['Age'] <= 13.333, '13.333'] = 1
    dataset.loc[(dataset['Age'] <= 26.667) & (dataset['Age'] > 13.333 ), '26.667'] = 1
    dataset.loc[(dataset['Age'] <= 40.0) & (dataset['Age'] > 26.667 ), '40.0'] = 1
    dataset.loc[(dataset['Age'] <= 53.333) & (dataset['Age'] > 40.0), '53.333'] = 1
    dataset.loc[(dataset['Age'] <= 66.667) & (dataset['Age'] > 53.333), '66.667'] = 1
    dataset.loc[dataset['Age'] > 66.667, '66.667+'] = 1

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
'''
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    for i in range(1):
        name = 'FamilySize' + str(i+1)
        dataset[name] = 0
        dataset.loc[dataset['FamilySize'] == i+1, name] = 1
    name = 'FamilySize>1'
    dataset[name] = 0
    dataset.loc[dataset['FamilySize'] >= 6, name] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
#print(train_df.head())


freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
for dataset in combine:
    for i in ['S', 'C', 'Q']:
        name = 'Embarked' + i
        dataset[name] = 0
        dataset.loc[dataset['Embarked'] == i, name] = 1

train_df = train_df.drop(['Embarked'], axis=1)
test_df = test_df.drop(['Embarked'], axis=1)
combine = [train_df, test_df]

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
'''
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    for i in range(4):
        dataset['Fare' + str(i+1)] = 0

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare1'] = 1
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare2'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare3'] = 1
    dataset.loc[dataset['Fare'] > 31, 'Fare4'] = 1

train_df = train_df.drop(['FareBand'], axis=1)
train_df = train_df.drop(['Fare'], axis=1)
test_df = test_df.drop(['Fare'], axis=1)
combine = [train_df, test_df]'''

for dataset in combine:
    for i in range(1,4):
        name = 'Pclass' + str(i)
        dataset[name] = 0
        dataset.loc[dataset['Pclass'] == i, name] = 1

for dataset in combine:
    dataset['Age'] = dataset['Age']/80
    dataset['Fare'] = dataset['Fare'] / 200

train_df = train_df.drop(['Pclass'], axis=1)
test_df = test_df.drop(['Pclass'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

train_set_frac = 0.8
path_train_input = r'data\train.csv'
path_my_train = Path(r'data\my_train3.csv')
path_my_val = Path(r'data\my_val3.csv')
path_my_test = Path(r'data\my_test3.csv')

train_data, val_data = train_test_split(train_df, test_size=1-train_set_frac)
print(len(val_data))
print(len(train_data))

path_my_train.parent.mkdir(parents=True, exist_ok=True)
path_my_val.parent.mkdir(parents=True, exist_ok=True)
path_my_test.parent.mkdir(parents=True, exist_ok=True)
train_data.to_csv(path_my_train, index=False)
val_data.to_csv(path_my_val, index=False)
test_df.to_csv(path_my_test, index=False)

