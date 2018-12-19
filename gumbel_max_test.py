import numpy as np
import matplotlib.pyplot as plt

N = 100000

M = 5

probs = np.random.rand(10)
probs = probs / np.sum(probs)

for i in range(M):
	results = []
	for j in range(N):
		noises = np.random.gumbel(loc=0.0, scale=1.0, size=(10,))

		z = np.argmax(np.log(probs) + noises)

		results.append(z)

	plt.hist(results)
	plt.show()

