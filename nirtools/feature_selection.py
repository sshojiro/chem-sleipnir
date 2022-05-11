import random
from copy import deepcopy
from deap import base, creator, tools
import logging
logging.basicConfig(format='%(name)s:%(level)s [%(asctime)s]| %(message)s',
    level=logging.DEBUG)
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm
from nirtools.utils import translate, generate_start_width, evaluate

def init_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    return creator

def create_toolbox(creator, max_width, n_features, n_regions, X, y, base_model):
    toolbox = base.Toolbox()
    toolbox.register("attribute", lambda:generate_start_width(n_features, max_width))
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attribute, n=n_regions)
    toolbox.register("population", tools.initRepeat, list,
                    toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    # skip mutate
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda individual:evaluate(individual, X, y, base_model))
    return toolbox

def gawls(toolbox, n_populations=100, cxpb=0.5, n_generations=50):
    """Genetic algorithm for wavelength selection (GAWLS)
    """
    pop = toolbox.population(n=n_populations)
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    for _ in tqdm(range(n_generations)):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # cross over
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # skip mutation
        invalid_ind = [ind for ind in offspring
                       if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
    return pop

class GAWLS(BaseEstimator):
    def __init__(self, base_model, max_width=10, n_regions=3,
        n_populations=100, n_generations=50):
        self.base_model = base_model
        self.best_model_ = deepcopy(base_model)
        self.n_regions = n_regions
        self.max_width = max_width
        self.n_populations = n_populations
        self.n_generations = n_generations
    def fit(self, X, y):
        creator_ = init_creator()
        self.n_features = X.shape[1]
        toolbox_ = create_toolbox(creator_, self.max_width, self.n_features,
            self.n_regions, X, y, self.base_model)
        self.pop = gawls(toolbox_, n_populations=self.n_populations,
            cxpb=0.5, n_generations=self.n_generations)
        fitnesses_ = np.array(list(map(toolbox_.evaluate, self.pop)))
        logging.debug(f'Population: {self.pop}')
        logging.debug('Best population: {}'.format(self.pop[fitnesses_.argmax()]))
        self.best_pop_ = self.pop[fitnesses_.argmax()]
        self.best_params_ = translate(self.best_pop_, self.n_features)
        self.best_model_.fit(X[:,self.best_params_], y)
        return self
    def predict(self, data):
        return self.base_model.predict(data[:,self.best_params_])