import torch
import random
import numpy as np
from collections import deque
from Snake_Game import Game, Coords
from Trainer import Net, Trainer
from matplotlib import pyplot as plt

plt.ion()

MEMORY_SIZE = 100000
SAMPLE_SIZE = 1000
LEARNING_RATE = 0.001


def manhattan_distance(point1, point2):
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)


class Snake:

    def __init__(self):
        self.num_game = 0
        self.epsilon = 0
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.model = Net(13, 256, 3)
        self.trainer = Trainer(self.model, learning_rate=LEARNING_RATE, gamma=0.9)

    def get_state(self, game):
        head = game.snake[0]
        left_wall = Coords(head.x - 20, head.y)
        right_wall = Coords(head.x + 20, head.y)
        up_wall = Coords(head.x, head.y - 20)
        down_wall = Coords(head.x, head.y + 20)

        left = game.direction == "LEFT"
        right = game.direction == "RIGHT"
        up = game.direction == "UP"
        down = game.direction == "DOWN"

        body_distance = [manhattan_distance(head, segment) for segment in game.snake[1:]]
        fruit_distance = manhattan_distance(game.head, game.fruit)

        close_to_body = 1 if any(distance <= 20 for distance in body_distance) else 0
        close_to_fruit = 1 if fruit_distance <= 60 else 0

        state = [
            # Collision with walls
            (right and game.check_collision(right_wall)) or
            (left and game.check_collision(left_wall)) or
            (up and game.check_collision(up_wall)) or
            (down and game.check_collision(down_wall)),

            (up and game.check_collision(right_wall)) or
            (down and game.check_collision(left_wall)) or
            (left and game.check_collision(up_wall)) or
            (right and game.check_collision(down_wall)),

            (down and game.check_collision(right_wall)) or
            (up and game.check_collision(left_wall)) or
            (right and game.check_collision(up_wall)) or
            (left and game.check_collision(down_wall)),

            # Current direction
            left,
            right,
            up,
            down,

            # Fruit location
            game.fruit.x < game.head.x,
            game.fruit.x > game.head.x,
            game.fruit.y < game.head.y,
            game.fruit.y > game.head.y,

            # Distance between snake's body and head
            close_to_body,

            # Distance between snake's head and fruit
            close_to_fruit
        ]

        return np.array(state, dtype=int)

    def memorize(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def training(self):
        if len(self.memory) > SAMPLE_SIZE:
            sample = random.sample(self.memory, SAMPLE_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def learn(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.num_game
        action = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action

    def mutate(self):
        # Mutation parameters needed for GA
        mutation_strength = 0.1
        # Mutate model parameters
        for param in self.model.parameters():
            mutation = random.uniform(-mutation_strength, mutation_strength)
            param.data += mutation

        return self


def plot(score, avg_score):
    plt.clf()
    plt.title('Snake 2, Model: Q-Learning (Reinforcement Learning)')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(score, label='Scores', color='blue')
    plt.plot(avg_score, label='Avg Scores', color='red')
    plt.ylim(ymin=0)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)


def train_snake():
    scores = []
    avg_score = []
    high_score = 0
    snake = Snake()
    game = Game()
    while True:

        prev_state = snake.get_state(game)
        action = snake.get_action(prev_state)

        reward, game_over, score = game.gameplay(action)
        state_new = snake.get_state(game)
        snake.learn(prev_state, action, reward, state_new, game_over)

        snake.memorize(prev_state, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            snake.num_game += 1
            snake.training()

            if score > high_score:
                high_score = score

            print('Game:', snake.num_game, 'Score:', score, 'High Score:', high_score)

            scores.append(score)
            avg = sum(scores) / len(scores)
            avg_score.append(avg)
            plot(scores, avg_score)


if __name__ == '__main__':
    train_snake()
