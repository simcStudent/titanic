import pandas as pd
from pathlib import Path

train_set_frac = 0.7
# TODO plot age against survive
path_train_input = r'data\train.csv'
path_my_train = Path(r'data\my_train2.csv')
path_my_val = Path(r'data\my_val2.csv')

pd.set_option('display.max_columns', None)
input_data = pd.read_csv(path_train_input)
print(input_data)
input_data = input_data.reindex(columns = ['Survived', 'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
input_data.insert(4, "firstClass", 0)
input_data.insert(5, "secondClass", 0)
input_data.insert(6, "thirdClass", 0)
input_data.insert(7, "under15", 0)
input_data.insert(7, "over55", 0)

input_data.loc[input_data['Pclass'] == 1, 'firstClass'] = 1
input_data.loc[input_data['Pclass'] == 2, 'secondClass'] = 1
input_data.loc[input_data['Pclass'] == 3, 'thirdClass'] = 1
input_data.loc[input_data['Age'] < 15, 'under15'] = 1
input_data.loc[input_data['Age'] > 55, 'over55'] = 1

input_data = input_data.reindex(columns = ['Survived', 'PassengerId', 'firstClass', 'secondClass', 'thirdClass', 'Sex',
                                           'under15', 'over55', 'SibSp', 'Parch', 'Fare'])

input_data.loc[input_data['Sex'] == 'male', 'Sex'] = 1
input_data.loc[input_data['Sex'] == 'female', 'Sex'] = 0
input_data = input_data.dropna()

def normalize(df_feature):
    max_value = df_feature.max()
    min_value = df_feature.min()
    result = (df_feature - min_value) / (max_value - min_value)
    return result

input_data['Fare'] = normalize(input_data['Fare'])

train_data = input_data.sample(frac=train_set_frac)
val_data = input_data.loc[~input_data.index.isin(train_data)]

path_my_train.parent.mkdir(parents=True, exist_ok=True)
path_my_val.parent.mkdir(parents=True, exist_ok=True)
train_data.to_csv(path_my_train, index=False)
val_data.to_csv(path_my_val, index=False)

