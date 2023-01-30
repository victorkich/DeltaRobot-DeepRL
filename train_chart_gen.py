import matplotlib.pyplot as plt
import numpy as np

print('Training charts:')
fig, ax = plt.subplots(1)
t = np.arange(200)
ax.plot(t, np.load('DQN_data.npy'), label="DQN")
ax.plot(t, np.load('DDQN_data.npy'), label="DDQN")
ax.plot(t, np.load('TRPO_data.npy'), label="TRPO")
ax.set_title("Train - Rewards vs Episodes")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend()
plt.savefig("training_charts.pdf", format="pdf", bbox_inches="tight", backend='pgf')
ax.cla()
plt.close(fig)
