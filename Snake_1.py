import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

pygame.init()

# dimensions of the screen
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

clock = pygame.time.Clock()


# Snake class
class Snake:
    def __init__(self):
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self.grow = False

    def move(self):
        head = self.body[0]
        x, y = head

        if self.direction == "UP":
            new_head = (x, y - 1)
        elif self.direction == "DOWN":
            new_head = (x, y + 1)
        elif self.direction == "LEFT":
            new_head = (x - 1, y)
        elif self.direction == "RIGHT":
            new_head = (x + 1, y)

        self.body.insert(0, new_head)
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

    def grow_snake(self):
        self.grow = True

    def draw(self):
        for segment in self.body:
            x, y = segment
            pygame.draw.rect(screen, GREEN, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    def check_collision(self):
        head = self.body[0]
        x, y = head
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        for segment in self.body[1:]:
            if segment == head:
                return True
        return False


# Fruit class
class Fruit:
    def __init__(self):
        self.position = self.new_position()

    def new_position(self):
        return random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)

    def draw(self):
        x, y = self.position
        pygame.draw.rect(screen, RED, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))


class Bot:
    def __init__(self, snake, fruit):
        self.snake = snake
        self.fruit = fruit
        self.width = WIDTH
        self.height = HEIGHT

    # move() based on the position of the fruit
    def move(self):
        head_x, head_y = self.snake.body[0]
        fruit_x, fruit_y = self.fruit.position
        dx = fruit_x - head_x
        dy = fruit_y - head_y

        if abs(dx) > abs(dy):
            if dx > 0 and self.snake.direction != "LEFT":
                return "RIGHT"
            elif dx < 0 and self.snake.direction != "RIGHT":
                return "LEFT"
        else:
            if dy > 0 and self.snake.direction != "UP":
                return "DOWN"
            elif dy < 0 and self.snake.direction != "DOWN":
                return "UP"
        # If no valid direction is found, continue with a random direction
        self.snake.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        return self.snake.direction

    #move_() based on the position of the fruit and avoids walls
    def move_2(self):
        head_x, head_y = self.snake.body[0]
        fruit_x, fruit_y = self.fruit.position

        # Calculate the difference between snake head and fruit position
        dx = fruit_x - head_x
        dy = fruit_y - head_y

        # Decide which direction to move based on the position of the fruit
        if abs(dx) > abs(dy):
            if dx > 0 and self.snake.direction != "LEFT":
                if head_x + 1 < self.width and (head_x + 1, head_y) not in self.snake.body:
                    return "RIGHT"
            elif dx < 0 and self.snake.direction != "RIGHT":
                if head_x - 1 >= 0 and (head_x - 1, head_y) not in self.snake.body:
                    return "LEFT"
        else:
            if dy > 0 and self.snake.direction != "UP":
                if head_y + 1 < self.height and (head_x, head_y + 1) not in self.snake.body:
                    return "DOWN"
            elif dy < 0 and self.snake.direction != "DOWN":
                if head_y - 1 >= 0 and (head_x, head_y - 1) not in self.snake.body:
                    return "UP"

        # if there's no fruit in sight, continue with a random direction that avoids the body or wall
        valid_directions = []
        if head_x + 1 < self.width and (head_x + 1, head_y) not in self.snake.body:
            valid_directions.append("RIGHT")
        if head_x - 1 >= 0 and (head_x - 1, head_y) not in self.snake.body:
            valid_directions.append("LEFT")
        if head_y + 1 < self.height and (head_x, head_y + 1) not in self.snake.body:
            valid_directions.append("DOWN")
        if head_y - 1 >= 0 and (head_x, head_y - 1) not in self.snake.body:
            valid_directions.append("UP")

        if valid_directions:
            return random.choice(valid_directions)
        else:
            # If no valid direction is found, continue with the current direction
            return self.snake.direction

#********************************************************************************************************
#Game with SnakeBot
def play_game(num_games):
    snake = Snake()
    fruit = Fruit()
    sbot = Bot(snake, fruit)
    score = 0

    font = pygame.font.SysFont('timesnewroman', 18)

    running = True
    while running:
        screen.fill(BLACK)

        #fruit position based move
        #snake.direction = sbot.move()
        #fruit finding and walls avoiding
        snake.direction = sbot.move_2()
        snake.move()
        if snake.body[0] == fruit.position:
            snake.grow_snake()
            score += 1
            fruit.position = fruit.new_position()

        if snake.check_collision():
            running = False

        text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(text, [0, 0])

        snake.draw()
        fruit.draw()
        pygame.display.flip()
        clock.tick(50)

    return sbot, running, score


def plot(score, avg_score):
    plt.clf()
    plt.title('Snake 1')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(score, label='Scores', color='blue')
    plt.plot(avg_score, label='Avg Scores', color='red')
    plt.ylim(ymin=0)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)


def main():
    scores = []
    avg_scores = []
    num_games = 0
    play = True
    high_score = 0
    while play:
        num_games += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False

        sbot, run, score = play_game(num_games)
        if not run:
            if score > high_score:
                high_score = score
            print('Game:', num_games, 'Score:', score, 'High Score:', high_score)
            scores.append(score)
            avg = sum(scores) / len(scores)
            avg_scores.append(avg)
            plot(scores, avg_scores)
    pygame.quit()


if __name__ == "__main__":
    main()
#********************************************************************************************************
#********************************************************************************************************
#human play mode
'''
def main():
    snake = Snake()
    fruit = Fruit()
    running = True

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction != "DOWN":
                    snake.direction = "UP"
                elif event.key == pygame.K_DOWN and snake.direction != "UP":
                    snake.direction = "DOWN"
                elif event.key == pygame.K_LEFT and snake.direction != "RIGHT":
                    snake.direction = "LEFT"
                elif event.key == pygame.K_RIGHT and snake.direction != "LEFT":
                    snake.direction = "RIGHT"

        snake.move()
        if snake.body[0] == fruit.position:
            snake.grow_snake()
            fruit.position = fruit.new_position()

        if snake.check_collision():
            running = False

        snake.draw()
        fruit.draw()

        pygame.display.flip()
        clock.tick(5)

    pygame.quit()

if __name__ == "__main__":
    main()
'''
#********************************************************************************************************
