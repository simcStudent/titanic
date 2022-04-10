import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pathlib import Path

batch_size = 240
epochs = 50
lr = 0.001
load_model = False
save_model = True
make_predictions = True

path_to_save_model_to = r'model3'
path_to_load_from = r'model3'
train_path = Path(r'data\my_train3.csv')
val_path = Path(r'data\my_val3.csv')
test_path = Path(f'data\my_test3.csv')
path_output = r'data\output\submission.csv'

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Running on the GPU')
else:
    device = torch.device("cpu")
    print('Running on the CPU')

#num_of_input_params = len(train_data.iloc[0])-1

class TitanicDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset.to(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        params = self.dataset[item, 1: self.dataset.size(1)]
        label = self.dataset[item, 0]
        return params, label

class TitanicDatasetTest(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset.to(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        params = self.dataset[item, 1: self.dataset.size(1)]
        passenger = self.dataset[item, 0]
        return params, passenger

training_data = np.loadtxt(str(train_path), dtype=np.float32, delimiter=",", skiprows=1)
training_data = torch.from_numpy(training_data)
train_data = TitanicDataset(training_data)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

validation_data = np.loadtxt(str(val_path), dtype=np.float32, delimiter=",", skiprows=1)
validation_dataset = torch.from_numpy(validation_data)
val_data = TitanicDataset(validation_dataset)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

test_data = np.loadtxt(str(test_path), dtype=np.float32, delimiter=",", skiprows=1)
test_dataset = torch.from_numpy(test_data)
test_data = TitanicDatasetTest(test_dataset)

num_input_params = training_data.size(1)-1
#print(f'training_data.size(0) {training_data.size(0)}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop = torch.nn.Dropout(0.5)
        self.input = torch.nn.Linear(num_input_params, 200)
        self.hidden1 = torch.nn.Linear(200, 200)
        self.hidden2 = torch.nn.Linear(200, 150)
        self.hidden3 = torch.nn.Linear(150, 100)
        self.hidden4 = torch.nn.Linear(100, 50)
        self.hidden5 = torch.nn.Linear(50, 50)
        self.hidden6 = torch.nn.Linear(50, 10)
        self.output = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.drop(x)
        x = torch.relu(self.hidden1(x))
        x = self.drop(x)
        x = torch.relu(self.hidden2(x))
        x = self.drop(x)
        x = torch.relu(self.hidden3(x))
        x = self.drop(x)
        x = torch.relu(self.hidden4(x))
        x = self.drop(x)
        x = torch.relu(self.hidden5(x))
        #x = self.drop(x)
        x = torch.relu(self.hidden6(x))
        #x = self.drop(x)
        x = torch.sigmoid(self.output(x)) #TODO sigmoid?
        return x

net = Net()
if load_model:
    net.load_state_dict(torch.load(path_to_load_from))

# TODO get CrossEntropyLoss to work
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
net.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        inputs, labels = data
        #print(f'inputs {inputs}')
        output = net(inputs)
        output = torch.squeeze(output)
        #output = output.to(torch.float32)
        #labels = labels.to(torch.int64)

        #print(output)
        #print(labels)

        val_loss = loss_func(output, labels)
        epoch_loss += val_loss.item()
        val_loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'epoch {epoch} loss = {epoch_loss}')

net.eval()
def accuracy(model, data):
    correct = 0
    total = 0
    for params, label in data:
        with torch.no_grad():
            output = model(params)
        if abs(label.item()-output.item()) < 0.5:
            correct += 1
        total += 1
    acc = correct/total
    return acc

acc_val = accuracy(net, val_data)
print(f'accuracy val {acc_val}')
print(f'accuracy train {accuracy(net, train_data)}')

if save_model:
    print('Saving model')
    torch.save(net.state_dict(), path_to_save_model_to)

if make_predictions:
    predictions = np.zeros(0).astype(int)
    PassengerId = np.zeros(0).astype(int)
    for params, passenger in test_data:
        output = net(params)
        output = torch.squeeze(output)
        #print(output.item())
        if output >= 0.5:
            predictions = np.append(predictions, 1)
        else:
            predictions = np.append(predictions, 0)
        PassengerId = np.append(PassengerId, int(passenger))

    submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": predictions
    })
    # print(submission)
    submission.to_csv(path_output, index=False)
