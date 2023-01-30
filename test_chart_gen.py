import numpy as np


def counter(data):
    positives = 0
    negatives = 0
    zeros = 0
    for d in data:
        if d > 0:
            positives += 1
        elif d < 0:
            negatives += 1
        else:
            zeros += 1
    return negatives, zeros, positives


dqn_charts = np.load('dqn_charts.npy')
ddqn_charts = np.load('ddqn_charts.npy')
trpo_charts = np.load('trpo_charts.npy')

print('DQN test data:')
print('Mean:', dqn_charts.mean())
print('STD:', dqn_charts.std())
dqn_negatives, dqn_zeros, dqn_positives = counter(dqn_charts)
print('Positives:', dqn_positives)
print('Zeros:', dqn_zeros)
print('Negatives:', dqn_negatives)

print('DDQN test data:')
print('Mean:', ddqn_charts.mean())
print('STD:', ddqn_charts.std())
ddqn_negatives, ddqn_zeros, ddqn_positives = counter(ddqn_charts)
print('Positives:', ddqn_positives)
print('Zeros:', ddqn_zeros)
print('Negatives:', ddqn_negatives)

print('TRPO test data:')
print('Mean:', trpo_charts.mean())
print('STD:', trpo_charts.std())
trpo_negatives, trpo_zeros, trpo_positives = counter(trpo_charts)
print('Positives:', trpo_positives)
print('Zeros:', trpo_zeros)
print('Negatives:', trpo_negatives)
