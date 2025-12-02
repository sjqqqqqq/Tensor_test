using Evolutionary

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2 # Rosenbrock
x0 = [0.0, 0.0];
Evolutionary.optimize(f, x0, CMAES(), Evolutionary.Options(iterations=10))