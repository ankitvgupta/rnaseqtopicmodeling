import numpy as np

# "maps a natural parameterization of the topic proportions to the mean parameterization"
# Equation 1 in 2007 paper
def f(eta):
	return np.exp(eta)/np.sum(eta)




