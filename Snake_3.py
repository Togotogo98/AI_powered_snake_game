from Snake_Game import Game
from Snake_2 import Snake
import random
import matplotlib.pyplot as plt

GAME_NUM = 800
POPULATION_SIZE = 50


def initialise_individual():
    snake = Snake()
    game = Game()
    high_score = 0
    scores = []
    avg_score = []
    while True:
        # each individual plays 200 games
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
            #here high score is used for evaluating fitness
            fitness = high_score

            print("Game:", snake.num_game, "Score:", score)

            scores.append(score)
            avg = sum(scores) / len(scores)
            avg_score.append(avg)
            plot_scores(scores, avg_score)

            if snake.num_game >= GAME_NUM:
                print("Total Games:", snake.num_game, "Highest Score:", high_score)
                individual = [snake, fitness]
                return individual


def initialise_population():
    population = []
    for i in range(POPULATION_SIZE):
        print("==========Snake individual:", i, "===========")
        individual = initialise_individual()
        population.append(individual)
    return population


def tournament_selection(population):
    refined_list = [[individual_snake, fitness] for individual_snake, fitness in population if fitness != 0]
    tournament_size = 5
    required = int(len(refined_list) * 0.8)
    selected_individuals = []

    # Perform tournament selection until we have at least the required number of selected individuals
    while len(selected_individuals) < required:
        # select individuals for the tournament
        tournament_candidates = random.sample(refined_list, tournament_size)

        # individual with the best fitness in the tournament selected and added to selected list
        best_individual = max(tournament_candidates, key=lambda x: x[1])
        selected_individuals.append(best_individual[0])

    return selected_individuals[:required]


def mutation(individual):
    mutation_rate = 0.1
    if random.random() < mutation_rate:
        mutated_individual = individual.mutate()
        return mutated_individual
    else:
        return individual


def crossover(parent1, parent2):
    offspring = Snake()
    for param_offspring, param_parent1, param_parent2 in zip(offspring.model.parameters(), parent1.model.parameters(),
                                                             parent2.model.parameters()):
        crossover_point = random.random()  # Randomly choose a crossover point
        param_offspring.data = crossover_point * param_parent1.data + (1 - crossover_point) * param_parent2.data

    return offspring


def evolve_population(population):
    #get a tournament selection
    selected_individuals = tournament_selection(population)
    new_population = []

    # Perform crossover and mutation to generate offspring
    while len(new_population) < POPULATION_SIZE:
        parent1 = random.choice(selected_individuals)
        parent2 = random.choice(selected_individuals)
        offspring = crossover(parent1, parent2)
        offspring = mutation(offspring)
        new_population.append(offspring)

    return new_population


def plot_scores(scores, avg_score):
    plt.clf()
    plt.title('Snake 3, Model: Genetic Algorithm')
    plt.xlabel('Number of Snakes')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores', color='blue')
    plt.plot(avg_score, label='Avg Scores', color='red')
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)


num_generations = 10
for generation in range(num_generations):
    print('Generation :', generation)
    # Evaluate fitness of the population
    population = initialise_population()

    # Evolve the population
    new_population = evolve_population(population)

