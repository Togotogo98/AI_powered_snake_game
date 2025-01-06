import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.input_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_hidden(x))
        x = self.hidden_output(x)
        return x


class Trainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        if all(var is not None for var in [state, action, reward, next_state, game_over]):
            state = torch.tensor(state).float()
            next_state = torch.tensor(next_state).float()
            action = torch.tensor(action).float()
            reward = torch.tensor(reward).float()

            if len(state.shape) == 1:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                game_over = (game_over, )

            predict = self.model(state)
            target = predict.clone()
            for i in range(len(game_over)):
                Q = reward[i]
                if not game_over[i]:
                    Q = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
                    target[i][torch.argmax(action).item()] = Q

            self.optimizer.zero_grad()
            ls = self.loss(target, predict)
            ls.backward()

            self.optimizer.step()








