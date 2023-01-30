import random
import torch as T
from gym import spaces
import math as mt
import matplotlib.pyplot as plt
import numpy as np


class DeltaEnv:
    def __init__(self):

        self.min_theta = -1.1
        self.max_theta = 1.1
        self.theta = np.array([0.0, 0.0, 0.0])

        self.distance = 1000
        self.current_distance = 1000

        self.set_increment_rate(0.1)
        self.render_counter = 0

        self.action = {
            0: "HOLD",
            1: "INC_J1",
            2: "DEC_J1",
            3: "INC_J2",
            4: "DEC_J2",
            5: "INC_J3",
            6: "DEC_J3",
            7: "INC_J1_J2",
            8: "DEC_J1_J2",
            9: "INC_J2_J3",
            10: "DEC_J2_J3",
            11: "INC_J1_J3",
            12: "DEC_J1_J3",
            13: "INC_J1_J2_J3",
            14: "DEC_J1_J2_J3"}

        self.ee_pos = self.forward_kinematics(
            self.theta[0], self.theta[1], self.theta[2])[0]
        self.goal_pos = self.generate_random_positions()[0]
        self.states = np.hstack(
            (self.goal_pos, self.ee_pos, self.theta, self.distance, 0, 0))

        self.action_space = spaces.Discrete(len(self.action))
        self.observation_space = spaces.Discrete(len(self.states))

    def forward_kinematics(self, t1, t2, t3):
        '''
            Esta função recebe como entrada angulos: theta_1, theta_2, theta_3 e
            retorna o ponto no espaço no qual o efetuador do robô deve se
            posicionar.
        '''
        X, Y, Z = 0, 0, 0

        L = 400  # mm
        l = 900
        rA = 180
        rE = 100

        # t1 = np.deg2rad(t1)
        # t2 = np.deg2rad(t2)
        # t3 = np.deg2rad(t3)

        phi = np.deg2rad(30)
        r = rA - rE

        x1 = 0
        y1 = - (r + L * mt.cos(t1))
        z1 = - L * mt.sin(t1)
        ponto_1 = np.array((x1, y1, z1))

        x2 = (r + L * mt.cos(t2)) * mt.cos(phi)
        y2 = (r + L * mt.cos(t2)) * mt.sin(phi)
        z2 = - L * mt.sin(t2)
        ponto_2 = np.array((x2, y2, z2))

        x3 = - (r + L * mt.cos(t3)) * mt.cos(phi)
        y3 = (r + L * mt.cos(t3)) * mt.sin(phi)
        z3 = - L * mt.sin(t3)
        ponto_3 = np.array((x3, y3, z3))

        p1 = y1**2 + z1**2
        p2 = x2**2 + y2**2 + z2**2
        p3 = x3**2 + y3**2 + z3**2

        a1 = (z2 - z1) * (y3 - y1) - (z3 - z1) * (y2 - y1)
        b1 = - ((p2 - p1) * (y3 - y1) - (p3 - p1) * (y2 - y1)) / 2

        a2 = - (z2 - z1) * x3 + (z3 - z1) * x2
        b2 = ((p2 - p1) * x3 - (p3 - p1) * x2) / 2

        dnm = (y2 - y1) * x3 - (y3 - y1) * x2

        a = a1**2 + a2**2 + dnm**2
        b = 2 * (a1 * b1 + a2 * (b2 - y1 * dnm) - z1 * dnm**2)
        c = (b2 - y1 * dnm) * (b2 - y1 * dnm) + \
            b1**2 + (dnm**2) * (z1**2 - l**2)

        d = b * b - 4.0 * a * c

        if (d < 0):
            Z = -1
            b + mt.sqrt(-d)
            b
            a
        else:
            Z = - 0.5 * (b + mt.sqrt(d)) / a
            X = (a1 * Z + b1) / dnm
            Y = (a2 * Z + b2) / dnm

        ee_pos = np.array((X, Y, Z))

        return ee_pos, ponto_1, ponto_2, ponto_3

    def set_increment_rate(self, rate):
        self.rate = rate

    def step(self, action):
        if self.action[action] == "HOLD":
            self.theta[0] += 0  # self.rate
            self.theta[1] += 0  # self.rate
            self.theta[2] += 0  # self.rate
        elif self.action[action] == "INC_J1":
            self.theta[0] += self.rate
        elif self.action[action] == "DEC_J1":
            self.theta[0] -= self.rate
        elif self.action[action] == "INC_J2":
            self.theta[1] += self.rate
        elif self.action[action] == "DEC_J2":
            self.theta[1] -= self.rate
        elif self.action[action] == "INC_J3":
            self.theta[2] += self.rate
        elif self.action[action] == "DEC_J3":
            self.theta[2] -= self.rate
        elif self.action[action] == "INC_J1_J2":
            self.theta[0] += self.rate
            self.theta[1] += self.rate
        elif self.action[action] == "DEC_J1_J2":
            self.theta[0] -= self.rate
            self.theta[1] -= self.rate
        elif self.action[action] == "INC_J2_J3":
            self.theta[1] += self.rate
            self.theta[2] += self.rate
        elif self.action[action] == "DEC_J2_J3":
            self.theta[1] -= self.rate
            self.theta[2] -= self.rate
        elif self.action[action] == "INC_J1_J3":
            self.theta[0] += self.rate
            self.theta[2] += self.rate
        elif self.action[action] == "DEC_J1_J3":
            self.theta[0] -= self.rate
            self.theta[2] -= self.rate
        elif self.action[action] == "INC_J1_J2_J3":
            self.theta[0] += self.rate
            self.theta[1] += self.rate
            self.theta[2] += self.rate
        elif self.action[action] == "DEC_J1_J2_J3":
            self.theta[0] -= self.rate
            self.theta[1] -= self.rate
            self.theta[2] -= self.rate

        self.ee_pos = self.forward_kinematics(
            self.theta[0], self.theta[1], self.theta[2])[0]

        self.theta[0] = np.clip(self.theta[0], self.min_theta, self.max_theta)
        self.theta[1] = np.clip(self.theta[1], self.min_theta, self.max_theta)
        self.theta[2] = np.clip(self.theta[2], self.min_theta, self.max_theta)

        self.theta[0] = self.normalize_angle(self.theta[0])
        self.theta[1] = self.normalize_angle(self.theta[1])
        self.theta[2] = self.normalize_angle(self.theta[2])

        a = np.array((self.goal[0], self.goal[1], self.goal[2]))
        b = np.array((self.ee_pos[0], self.ee_pos[1], self.ee_pos[2]))

        self.distance = np.linalg.norm(a-b)
        #print("DISTANCIA DELTA ENV: ", distance)

        done = False
        reward = 0
        if self.distance >= self.current_distance:
            reward = -1

        epsilon = 50

        if (self.distance > -epsilon and self.distance < epsilon):
            reward = 1
            done = True

        self.current_distance = self.distance
        self.current_score += reward

        if self.current_score == -10 or self.current_score >= 10:
            done = True
        else:
            done = False

        observation = np.hstack(
            (self.goal_pos, self.ee_pos, self.theta, self.distance))
        info = [
            self.distance,
            self.goal_pos,
            self.ee_pos
        ]
        #self.render(self.theta, self.goal_pos)
        # self.timerate.sleep()
        return observation, reward, done, info

    def generate_random_positions(self):
        angles = np.arange(-1.1, 1.1, 0.01).tolist()

        theta0 = random.choice(angles)
        theta1 = random.choice(angles)
        theta2 = random.choice(angles)
        degree_angles = [
        [-45, -45, -45], # 1
        [0, -45, -45], # 2
        [0, 0, -45], # 3
        [45, -45, 0], # 4
        [30, 30, 0], # 5
        [-45, 25, -25],
        [-10, 30, -30], # 7
        [45, 45, 45],
        [-30, -30, 0],
        [-45, 25, -25],
        [30, 0, 45],
        [15, -15, 25],
        [10,-25, -15],
        [-35, -15, 5],
        [10, -30, 10],
        [-25, 10, -5],
        [-30, 5, 30],
        [-30, 45, 30],
        [-15, 10, 15],
        [-5, -10, 5],
        [-10, 30, 40],
        [-30, -30, -30],]

        radian_angles = []
        rangle = []

        for angle in degree_angles:
            rangle.append([np.deg2rad(angle[0]), np.deg2rad(angle[1]), np.deg2rad(angle[2])])
        # print(rangle)
        # print(radian_angles)

        theta0, theta1, theta2 = np.choose(np.random.randint(0, len(degree_angles)), degree_angles)

        self.goal = self.forward_kinematics(theta0, theta1, theta2)[0]

        return self.goal

    @staticmethod
    def normalize_angle(angle):
        return mt.atan2(mt.sin(angle), mt.cos(angle))

    def reset(self, ep):
        self.goal_pos = self.generate_random_positions()

        self.theta[0] = 0
        self.theta[1] = 0
        self.theta[2] = 0

        self.current_score = 0

        self.ee_pos = self.forward_kinematics(self.theta[0], self.theta[1], self.theta[2])[0]
        observation = np.hstack((self.goal_pos, self.ee_pos, self.theta, self.distance))

        return observation

    def render(self, theta, goal_pos):
        data = self.forward_kinematics(theta[0], theta[1], theta[2])

        ee_pos = data[0]
        ponto_1 = data[1]
        ponto_2 = data[2]
        ponto_3 = data[3]
        fig = plt.figure(1, figsize=(5, 5))

        ax = fig.add_subplot(111, projection='3d')

        # plot the point (2,3,4) on the figure
        ax.scatter(ee_pos[0], ee_pos[1], ee_pos[2], c='black', marker='o')
        # plot the point (2,3,4) on the figure
        ax.scatter(ponto_1[0], ponto_1[1], ponto_1[2], c='blue', marker='$O$')
        # plot the point (2,3,4) on the figure
        ax.scatter(ponto_2[0], ponto_2[1], ponto_2[2], c='blue', marker='$O$')
        # plot the point (2,3,4) on the figure
        ax.scatter(ponto_3[0], ponto_3[1], ponto_3[2], c='blue', marker='$O$')
        ax.scatter(goal_pos[0], goal_pos[1],
                   goal_pos[2], c='red', marker='$X$')

        t = np.linspace(0, 2*np.pi, 1000)
        z_line = ee_pos[2]  # np.linspace(0, 15, 1000)
        x_line = ee_pos[0] + 100*np.cos(t)
        y_line = ee_pos[1] + 100*np.sin(t)

        z_line1 = 0  # np.linspace(0, 15, 1000)
        x_line1 = 180*np.cos(t)
        y_line1 = 180*np.sin(t)

        ax.plot3D(x_line, y_line, z_line, 'blue')
        ax.plot3D(x_line1, y_line1, z_line1, 'red')

        x = [0,    ponto_1[0], 180 *
             np.sin(np.deg2rad(60)), ponto_2[0], -180*np.sin(np.deg2rad(60)), ponto_3[0]]
        y = [-180,    ponto_1[1], 180 *
             np.cos(np.deg2rad(60)), ponto_2[1],  180*np.cos(np.deg2rad(60)), ponto_3[1]]
        z = [0,    ponto_1[2], 0, ponto_2[2],
             0,                        ponto_3[2]]

        x1 = [0 + ee_pos[0],    ponto_1[0], 100*np.sin(np.deg2rad(
            60)) + ee_pos[0], ponto_2[0], -100*np.sin(np.deg2rad(60)) + ee_pos[0], ponto_3[0]]
        y1 = [-100 + ee_pos[1],    ponto_1[1], 100*np.cos(np.deg2rad(
            60)) + ee_pos[1], ponto_2[1],  100*np.cos(np.deg2rad(60)) + ee_pos[1], ponto_3[1]]
        z1 = [ee_pos[2],    ponto_1[2],                              ee_pos[2],
              ponto_2[2],                               ee_pos[2], ponto_3[2]]

        plt.plot(x, y, z,  'ro')

        def connectpoints(x, y, z, p1, p2):
            x1, x2 = x[p1], x[p2]
            y1, y2 = y[p1], y[p2]
            z1, z2 = z[p1], z[p2]
            plt.plot([x1, x2], [y1, y2], [z1, z2], 'k-')

        connectpoints(x, y, z, 0, 1)
        connectpoints(x, y, z, 2, 3)
        connectpoints(x, y, z, 4, 5)

        connectpoints(x1, y1, z1, 0, 1)
        connectpoints(x1, y1, z1, 2, 3)
        connectpoints(x1, y1, z1, 4, 5)

        ax.view_init(elev=30, azim=60)

        ax.axes.set_xlim3d(left=-650, right=650)
        ax.axes.set_ylim3d(bottom=-650, top=650)
        ax.axes.set_zlim3d(bottom=-1200, top=100)

        self.render_counter += 1
        plt.savefig(
            f"{RENDERS_DIR}/step_{self.render_counter}")
        ax.cla()
        plt.close(fig)


import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, lr=0.00001, n_states=4, n_actions=6):
        super(QNetwork, self).__init__()
        # Detalhe da rede neural
        self.fc1 = nn.Linear(n_states, 300)
        self.fc2 = nn.Linear(300, 400)
        self.fc3 = nn.Linear(400, 600)
        self.fc4 = nn.Linear(600, 600)
        self.fc5 = nn.Linear(600, 400)
        self.fc6 = nn.Linear(400, 300)
        self.fc7 = nn.Linear(300, n_actions)
        # Optimizer e loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.fc7(x)

    def save_checkpoint(self):
        print('... Save checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... Load checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


def test_model(model):
    env = DeltaEnv()
    device = 'cuda'
    n_episodes = 100
    n_steps = 500

    total_reward_hist = []
    avg_reward_hist = []
    for episode in range(1, n_episodes + 1):
        state = env.reset(episode)
        total_reward = 0
        for t in range(n_steps):
            state = T.tensor(np.array([state]), dtype=T.float)  #.to(device)
            actions = model.forward(state)
            action = T.argmax(actions).item()
            next_state, reward, done, info = env.step(action)

            state = next_state
            total_reward += reward
            if done:
                break

        total_reward_hist.append(total_reward)
        avg_reward = np.average(total_reward_hist[-100:])
        avg_reward_hist.append(avg_reward)
        print("Episode :", episode, "Total Reward : {:.4f}".format(total_reward), "Avg Reward : {:.4f}".format(avg_reward))

        return total_reward_hist

"""
print('Training charts:')
fig, ax = plt.subplots(1)
t = np.arange(200)
ax.plot(t, np.load('DQN_data.npy'), label="DQN")
ax.plot(t, np.load('DDQN_data.npy'), label="DDQN")
ax.plot(t, np.load('TRPO_data.npy'), label="TRPO")
ax.set_title("TREINO - Recompensas vs Episódios")
ax.set_xlabel("Episódio")
ax.set_ylabel("Recompensa")
ax.legend()
plt.savefig("training_charts.png")
ax.cla()
plt.close(fig)
"""

print('Running tests...')
dqn_agent = T.load('DQN_agent.pt', map_location=T.device('cpu'))
dqn_agent.eval()
dqn_reward_hist = test_model(dqn_agent)
del(dqn_agent)
ddqn_agent = T.load('DDQN_agent.pt', map_location=T.device('cpu'))
ddqn_agent.eval()
ddqn_reward_hist = test_model(ddqn_agent)
del(ddqn_agent)
trpo_agent = T.load('TRPO_agent.pt', map_location=T.device('cpu'))
trpo_agent.eval()
trpo_reward_hist = test_model(trpo_agent)
del(trpo_agent)

print('Testing charts:')
fig, ax = plt.subplots(1)
t = np.arange(100)
ax.plot(t, dqn_reward_hist, label="DQN")
ax.plot(t, ddqn_reward_hist, label="DDQN")
ax.plot(t, trpo_reward_hist, label="TRPO")
ax.set_title("TESTE - Recompensas vs Episódios")
ax.set_xlabel("Episódio")
ax.set_ylabel("Recompensa")
ax.legend()
plt.savefig("testing_charts.png")
ax.cla()
plt.close(fig)
