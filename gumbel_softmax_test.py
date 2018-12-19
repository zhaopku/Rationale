import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

N = 1000000

probs = np.random.rand(3)
probs = probs / np.sum(probs)
t = 0.1

print(probs)

cnt = defaultdict(int)

results = []
for j in range(N):
	noises = np.random.gumbel(loc=0.0, scale=1.0, size=(3,))

	z = np.exp((np.log(probs) + noises)/t) / np.sum(np.exp((np.log(probs) + noises)/t))
	z = np.argmax(z)
	results.append(z)
	cnt[z] += 1

new_probs = []
for i in range(3):
	new_probs.append(cnt[i]/N)

print(new_probs)
