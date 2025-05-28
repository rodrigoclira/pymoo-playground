from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.population import Population
import numpy as np

problem = get_problem("g1")


X = np.random.random((100, problem.n_var))
pop = Population.new("X", X)

algorithm = GA(sampling=pop, eliminate_duplicates=True)

res = minimize(problem, algorithm, seed=1, verbose=True, save_history=True)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
