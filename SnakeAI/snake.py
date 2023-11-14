import pygame, random, math, time
from neural_network import *
pygame.init()

# GAME CONSTANTS

SIZE = 20
COLS, ROWS = 20, 20
WIDTH, HEIGHT = SIZE*COLS, SIZE*ROWS
FPS = 10

POP_SIZE = 100
MUTATION_PROB = 0.03
NN_SHAPE = [8, 10, 4]

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)

# FUNCTIONS

def fitness1(score, time_alive):
    return round((time_alive)/10) + (score*5)**2

def random_pos(taken_pos):
    '''Generates a random position on the game grid that isn't already taken by the snake.'''
    while True:
        x = random.randint(0,COLS-1)
        y = random.randint(0,ROWS-1)
        if not [x,y] in taken_pos:
            return [x,y]

class Snake:
    def __init__(self, neural):
        self.brain = neural
        self.length = 3
        self.score = 0
        self.fitness = 0
        self.player_parts = [[COLS//2-i, ROWS//2] for i in range(self.length)]
        self.velocity = [1,0]
        self.fruit = random_pos(self.player_parts)
        self.total_frames = 0
        self.frames_til_death = 50
        self.clock = pygame.time.Clock()
        self.gameover = False

# BRAIN METHODS

    def calculate_fitness(self, func=fitness1):
        return func(self.score, self.total_frames)

    def get_inputs(self):
        head = self.player_parts[0]
        headx, heady = head
        body = self.player_parts[1:]

        inputs = [headx, heady, self.fruit[0], self.fruit[1], 0, 0, 0, 0]
        if [headx, heady - 1] in body: # Checks for snake body in upwards direction
            inputs[4] = 1
        if [headx, heady + 1] in body: # Checks for snake body in downwards direction
            inputs[5] = 1
        if [headx - 1, heady] in body: # Checks for snake body in left direction
            inputs[6] = 1
        if [headx + 1, heady] in body: # Checks for snake body in right direction
            inputs[7] = 1

        return inputs

    def get_move(self, inputs):
        choices = self.brain.propagate_forward(inputs)
        direction = choices.index(max(choices))
        return direction
    
# PLAYER METHODS

    def turn_up(self):
        if not self.velocity == [0, 1]:
            self.velocity = [0, -1]

    def turn_down(self):
        if not self.velocity == [0, -1]:
            self.velocity = [0, 1]

    def turn_left(self):
        if not self.velocity == [1, 0]:
            self.velocity = [-1, 0]

    def turn_right(self):
        if not self.velocity == [-1, 0]:
            self.velocity = [1, 0]

    def move(self):
        '''Moves the snake based on the current velocity/direction.'''
        next_x = self.player_parts[0][0] + self.velocity[0] # Calculates the next x position for the snakes 
        next_y = self.player_parts[0][1] + self.velocity[1] # Calculates the next y position for the snakes head
        self.player_parts.pop() # Removes the last part of the snakes body 
        self.player_parts.insert(0, [next_x, next_y]) # Adds the new head position to the front of the snake

    def add(self):
        '''Adds a new segment to the back of the snake.'''
        new_pos = self.player_parts[-1] 
        self.player_parts.append(new_pos)

    def kill(self):
        self.gameover = True
        self.fitness = self.calculate_fitness(fitness1)

# GAME METHODS

    def change_direction(self):
        '''Changes the snake velocity/direction depending on what choice is made by the neural network.'''
        direction = self.get_move(self.get_inputs())
        if direction == 0:
            self.turn_up()
        elif direction == 1:
            self.turn_down()
        elif direction == 2:
            self.turn_left()
        elif direction == 3:
            self.turn_right()

    def fruit_collision(self):
        '''Adds to the length of the snake, resets the frames fruitless and generates a new fruit if the snake a fruit collide.'''
        if self.player_parts[0] == self.fruit:
            self.add()
            self.length += 1
            self.score += 1
            self.frames_til_death += 50
            self.fruit = random_pos(self.player_parts)

    def gameend_check(self):
        '''Sets gameover to True if the snake collides with itself or the edges of the screen.'''
        head = self.player_parts[0]
        headx, heady = head
        body = self.player_parts[1:]
        
        if head in body:
            self.kill()
        elif headx < 0 or headx >= COLS:
            self.kill()
        elif heady < 0 or heady >= ROWS:
            self.kill()
        elif self.frames_til_death == 0:
            self.kill()

    def update(self):
        '''Updates the snake each turn/move.'''
        self.total_frames += 1
        self.frames_til_death -= 1
        self.change_direction()
        self.move()
        self.fruit_collision()
        self.gameend_check()

    def fruit_render(self, screen):
        '''Renders the fruit on the screen.'''
        pygame.draw.rect(screen, RED, (SIZE*self.fruit[0], SIZE*self.fruit[1], SIZE, SIZE))

    def snake_render(self, screen):
        '''Renders the snake to the screen'''
        for part in self.player_parts:
            pygame.draw.rect(screen, WHITE, (SIZE*part[0], SIZE*part[1], SIZE, SIZE))

    def render(self, screen):
        '''Renders the game to the pygame screen.'''
        screen.fill(BLACK)
        self.snake_render(screen)
        self.fruit_render(screen)
        pygame.display.update()

    def run(self, screen):
        '''Runs the game until the snake dies.'''
        self.gameover = False
        while not self.gameover:
            self.render(screen)
            self.update()
            self.clock.tick(FPS)

class Population:
    def __init__(self, size, snakes):
        self.size = size
        self.snakes = snakes
        self.count = 0

    def update(self):
        while self.count < self.size:
            for snake in self.snakes:
                if not snake.gameover:
                    snake.update()
                    if snake.gameover:
                        self.count += 1
        return True

    def generate_new_population(self):
        mating_pool = []
        max_fitness = 0
        for snake in self.snakes:
            if snake.fitness > max_fitness:
                max_fitness = snake.fitness
                best_snake = snake.brain
        for snake in self.snakes:
            for i in range(math.floor((snake.fitness/max_fitness)*50)):
                mating_pool.append(snake)

        games = [Snake(best_snake)]

        for i in range(POP_SIZE-1):
            parent_a = random.choice(mating_pool).brain.flatten()
            parent_b = random.choice(mating_pool).brain.flatten()
            split_pos = random.randint(1, len(parent_a))
            combination = parent_a[:split_pos] + parent_b[split_pos:]
            for i in range(len(combination)):
                if random.random() <= MUTATION_PROB:
                    combination[i] = random.random()*2 - 1
            snake = NeuralNetwork.unflatten(NN_SHAPE, combination)
            games.append(Snake(snake))
        return Population(POP_SIZE, games), best_snake

    @classmethod
    def create_random_population(cls, size):
        new_pop = []
        for i in range(size):
            new_pop.append(Snake(NeuralNetwork.random(NN_SHAPE)))
        return Population(size, new_pop)

class Simulation:
    def __init__(self):
        self.population = Population.create_random_population(POP_SIZE)
        self.generation = 1
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.max_score = 0
        self.start_time = time.time()
        self.population_done = False

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    print("Generation:", self.generation, "Time:", time.time() - self.start_time, "Max Score:", self.max_score)

    def update(self):
        while not self.population_done:
            self.population_done = self.population.update()
            self.process_events()
        for snake in self.population.snakes:
            if snake.score > self.max_score:
                self.max_score = snake.score
                game = Snake(snake.brain)
                game.run(self.screen)
                        
    def run(self):
        while True:
            self.update()
            self.population, best_snake = self.population.generate_new_population()
            self.population_done = False
            if self.generation % 100 == 0:
                game = Snake(best_snake)
                game.run(self.screen)
            self.generation += 1
        
simulation = Simulation()
simulation.run()
            
