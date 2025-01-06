import pygame
import random
from collections import namedtuple
import numpy as np

pygame.init()

Coords = namedtuple('Point', 'x, y')


Font = pygame.font.SysFont('timesnewroman', 18)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

GRID_SIZE = 20


class Game:
    def __init__(self, width=600, height=400):
        self.w = width
        self.h = height
        # init display
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        self.head = Coords(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Coords(self.head.x - GRID_SIZE, self.head.y),
                      Coords(self.head.x - (2 * GRID_SIZE), self.head.y)]
        self.score = 0
        self.fruit = None
        self.new_fruit()
        self.game_iterations = 0

    def new_fruit(self):
        x = random.randint(0, (self.w - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
        y = random.randint(0, (self.h - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
        self.fruit = Coords(x, y)
        # to avoid generating fruit in the same position as the snake body:
        if self.fruit in self.snake:
            self.new_fruit()

    def gameplay(self, action):
        self.game_iterations += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.check_collision() or self.game_iterations > 100 * len(self.snake):
            game_over = True
            reward = -1
            return reward, game_over, self.score

        if self.head == self.fruit:
            self.score += 1
            reward = 1
            self.new_fruit()
        else:
            self.snake.pop()

        self.update()
        self.clock.tick(50)
        return reward, game_over, self.score

    def check_collision(self, point=None):
        if point is None:
            point = self.head
        if point.x > self.w - GRID_SIZE or point.x < 0 or point.y > self.h - GRID_SIZE or point.y < 0:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def update(self):
        self.screen.fill(BLACK)
        # draw the snake
        for pt in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(pt.x, pt.y, GRID_SIZE, GRID_SIZE))
        # draw the fruit
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.fruit.x, self.fruit.y, GRID_SIZE, GRID_SIZE))
        #display score
        text = Font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):

        #new_direction = self.direction
        # debug: print("action:", action)
        action_to_direction = {
            "UP": {0: "UP", 1: "LEFT", 2: "RIGHT"},
            "DOWN": {0: "DOWN", 1: "RIGHT", 2: "LEFT"},
            "LEFT": {0: "LEFT", 1: "DOWN", 2: "UP"},
            "RIGHT": {0: "RIGHT", 1: "UP", 2: "DOWN"}
        }

        # Get the current direction of the snake
        current_direction = self.direction
        # debug: print("current_direction", current_direction)

        # Get the corresponding direction based on the action and current direction
        new_direction = action_to_direction[current_direction][np.argmax(action)]
        # debug: print("new_direction", new_direction)
        # Update the head position based on the new direction
        x = self.head.x
        y = self.head.y
        if new_direction == "RIGHT":
            x += GRID_SIZE
        elif new_direction == "LEFT":
            x -= GRID_SIZE
        elif new_direction == "DOWN":
            y += GRID_SIZE
        elif new_direction == "UP":
            y -= GRID_SIZE

        self.head = Coords(x, y)
        self.direction = new_direction


#game by human
'''
def main():
    game = Game()
    # game loop
    while True:
        game_over, score = game.gameplay()
        if game_over:
            break


if __name__ == "__main__":
    main()
'''
