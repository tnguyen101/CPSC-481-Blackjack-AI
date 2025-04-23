import torch
import torch.nn as nn

class BlackJackModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BlackJackModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def load_model(path="blackjack_model.pth"):

    model = BlackJackModel(input_size=5, hidden_size=32, output_size=4)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model

def predict_action_from(game_state, model):

    action_map = {0: "hit", 1: "stand", 2: "double", 3: "surrender"}

    x = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        predicted = torch.argmax(output, dim=1).item()
        
    return action_map[predicted]