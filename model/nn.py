
import torch
import pandas as pd 
import numpy as np
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pdb
### Customized the dataset and dataloader
class EDdataset(Dataset):
    def __init__(self, ratio=0.8):
        super().__init__()
        df = pd.read_csv('./data/output_data.csv', index_col=0)
        df = df[~((df['Date'] >= '2020-03s-15') & (df['Date'] < '2020-05-14'))]
        df.reset_index(drop=True, inplace=True)
        # encoding the province
        df = pd.get_dummies(df, columns=['Province'], dtype=float)

        X = df.drop(columns=['Date','Number_Visits', 'holiday_name', 'normal day'])
        y = df['Number_Visits'].map(lambda x: int(x.replace(',', '')))
        self.X_df = X
        ct = ColumnTransformer([
        ('weathers scaler', StandardScaler(), ['MIN_TEMPERATURE', 'MEAN_TEMPERATURE', 'MAX_TEMPERATURE', 'TOTAL_SNOW',
       'TOTAL_RAIN', 'TOTAL_PRECIPITATION', 'HEATING_DEGREE_DAYS', 'COOLING_DEGREE_DAYS'])
    ], remainder='passthrough')
        split_index = int(ratio * len(X))
        X_train = X[:split_index]
        y_train = y[:split_index]
        ct_trans = ct.fit(X_train, y_train)
        X_transformed = ct_trans.transform(X)
        #### normalize the data
        # X = MinMaxScaler().fit_transform(X)
        assert X_transformed.shape[0] == len(y)
        # pdb.set_trace()
        self.X = torch.tensor(X_transformed, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    
    def indices_of_province(self, province, test_range):
        X_test = self.X_df.iloc[test_range[0]: test_range[1]]
        indices = X_test.index[X_test[province] == 1].tolist()
        return indices


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Define the neural network architecture
class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(23, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.network(x)

# Define the training function
def train(net, train_loader, optimizer, criterion):
    net.train()
    train_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = net(data).squeeze()
        loss = criterion(output, target)
        train_loss += loss.item()
        count = count + 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / count

# Define the testing function
def test(net, test_loader, criterion):
    net.eval()
    test_loss = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data).squeeze()
            test_loss += criterion(output, target).item()
            count = count + 1
    return test_loss / count

# Define the main function
def main():
    # Load the data
    ratio = 0.8
    dataset = EDdataset(ratio)
    num_epochs = 2000
    province = 'BC'

    ### train test split
    
    train_size = int(ratio * len(dataset))
    train_set = Subset(dataset, range(train_size))
    idx_range = dataset.indices_of_province(f'Province_{province}', [train_size, len(dataset)])
    ### filter the dataset so that the test set only contains the data from the province
    test_set_province = Subset(dataset, idx_range)
    train_dl = DataLoader(train_set, batch_size=512, shuffle=True)
    test_dl = DataLoader(test_set_province, batch_size=len(test_set_province), shuffle=False)
    # Initialize the neural network
    net = EDNet()

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train the neural network
    for epoch in range(num_epochs):
        train_loss = train(net, train_dl, optimizer, criterion)
        print('Epoch: {}, training loss: {:.4f}'.format(epoch, train_loss))

    # Test the neural network
    test_loss = test(net, test_dl, criterion)
    print('Test L2 Loss: {:.6f}'.format(test_loss))

    L1_loss = test(net, test_dl, nn.L1Loss())
    print('Test L1 Loss: {:.6f}'.format(L1_loss))

if __name__ == '__main__':
    main()
