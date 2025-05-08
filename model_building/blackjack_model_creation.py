from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import torch

class BlackjackDataset(Dataset):
  def __init__(self, dataset):
    self.X = dataset.drop(columns=['action']).values.astype('float32')
    self.y = dataset['action'].values.astype('int64')

  def __len__(self):
    return len(self.y)

  def __getitem__(self, index):
     return torch.tensor(self.X[index]), torch.tensor(self.y[index])

class BlackJackModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BlackJackModel, self).__init__()
        # input size as input, and hidden size as output
        self.l1 = nn.Linear(input_size, hidden_size)
        # actuation function
        self.relu = nn.ReLU()
        # hidden_size as input and output_size an output
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    


def create_blackjack_model():
    
    # reading CSV file
    blackjack_csv = pd.read_csv('blackjack_data.csv')

    # splitting into train and test data
    train_data, test_data = train_test_split(blackjack_csv, test_size=0.2, random_state=42)

    train_dataset = BlackjackDataset(train_data)
    test_dataset = BlackjackDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # hyperparameters
    input_size = 5
    hidden_size = 32
    output_size = 4

    # initalize model
    model = BlackJackModel(input_size, hidden_size, output_size)



    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    total_steps = len(train_loader)
    num_epochs = 10

    # loops through adjusting weights
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    # prints accuracy 
    with torch.no_grad():
        n_correct = 0
        n_samples = len(test_loader.dataset)

        for inputs, labels in test_loader:

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

        acc = n_correct / n_samples
        print(f'Accuracy of the network on the {n_samples} test inputs: {100*acc} %')

    # saves model in file "blackjack_model.pth"
    torch.save(model.state_dict(), "blackjack_model.pth")

if __name__ == "__main__":
    create_blackjack_model()


