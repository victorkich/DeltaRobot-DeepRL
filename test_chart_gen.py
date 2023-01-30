import numpy as np


def counter(data):
    positives = 0
    negatives = 0
    zeros = 0
    positives_list = list()
    negatives_list = list()
    zeros_list = list()
    for d in data:
        if d > 0:
            positives += 1
            positives_list.append(d)
        elif d < 0:
            negatives += 1
            negatives_list.append(d)
        else:
            zeros += 1
            zeros_list.append(d)

    positives_list = np.array(positives_list)
    negatives_list = np.array(negatives_list)
    zeros_list = np.array(zeros_list)
    if not len(positives_list):
        positives_list = np.zeros(1)
    if not len(negatives_list):
        negatives_list = np.zeros(1)
    if not len(zeros_list):
        zeros_list = np.zeros(1)
    return negatives, zeros, positives, positives_list.mean(), positives_list.std(), negatives_list.mean(), \
        negatives_list.std(), zeros_list.mean(), zeros_list.std()


dqn_charts = np.load('dqn_charts.npy')
ddqn_charts = np.load('ddqn_charts.npy')
trpo_charts = np.load('trpo_charts.npy')

print('DQN test data:')
print('Mean:', dqn_charts.mean())
print('STD:', dqn_charts.std())
dqn_negatives, dqn_zeros, dqn_positives, positive_mean, positive_std, negative_mean, negative_std, zeros_mean, zeros_std = counter(dqn_charts)
print('Positives:', dqn_positives, 'mean:', positive_mean, 'std:', positive_std)
print('Zeros:', dqn_zeros, 'mean:', zeros_mean, 'std:', zeros_std)
print('Negatives:', dqn_negatives, 'mean:', negative_mean, 'std:', negative_std)

print('DDQN test data:')
print('Mean:', ddqn_charts.mean())
print('STD:', ddqn_charts.std())
ddqn_negatives, ddqn_zeros, ddqn_positives, positive_mean, positive_std, negative_mean, negative_std, zeros_mean, zeros_std = counter(ddqn_charts)
print('Positives:', ddqn_positives, 'mean:', positive_mean, 'std:', positive_std)
print('Zeros:', ddqn_zeros, 'mean:', zeros_mean, 'std:', zeros_std)
print('Negatives:', ddqn_negatives, 'mean:', negative_mean, 'std:', negative_std)

print('TRPO test data:')
print('Mean:', trpo_charts.mean())
print('STD:', trpo_charts.std())
trpo_negatives, trpo_zeros, trpo_positives, positive_mean, positive_std, negative_mean, negative_std, zeros_mean, zeros_std = counter(trpo_charts)
print('Positives:', trpo_positives, 'mean:', positive_mean, 'std:', positive_std)
print('Zeros:', trpo_zeros, 'mean:', zeros_mean, 'std:', zeros_std)
print('Negatives:', trpo_negatives, 'mean:', negative_mean, 'std:', negative_std)
