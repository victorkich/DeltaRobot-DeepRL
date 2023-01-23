import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch as T
import random
import numpy as np
from gym import spaces
import math as mt
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0001
EPS = 0.1
EPS_DECAY = 0.999
EPS_MIN = 0.01
NUMBER_OF_EPISODES = 2302
EPISODE_TO_START_PRINTING = NUMBER_OF_EPISODES - 10


class QNetwork(nn.Module):
    def __init__(self, lr=LEARNING_RATE, n_states=4, n_actions=6):
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


class ReplayMemory(object):
    def __init__(self, max_size=10000, n_states=4):
        self.max_size = max_size
        self.memory_counter = 0
        self.states_memory = np.zeros((max_size, n_states), dtype=np.float32)
        self.next_states_memory = np.zeros(
            (max_size, n_states), dtype=np.float32)
        self.actions_memory = np.zeros(max_size, dtype=np.int64)
        self.rewards_memory = np.zeros(max_size, dtype=np.float32)
        self.dones_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.max_size
        self.states_memory[index] = state
        self.actions_memory[index] = action
        self.rewards_memory[index] = reward
        self.next_states_memory[index] = next_state
        self.dones_memory[index] = done
        self.memory_counter += 1

    def sample_memory(self, batch_size):
        max_mem = min(self.memory_counter, self.max_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.states_memory[batch]
        actions = self.actions_memory[batch]
        rewards = self.rewards_memory[batch]
        next_states = self.next_states_memory[batch]
        dones = self.dones_memory[batch]
        return states, actions, rewards, next_states, dones


class DQNAgent(object):
    def __init__(
        self,
        alpha=0.0005,
        gamma=0.99,
        eps=EPS,
        eps_decay=EPS_DECAY,
        eps_min=EPS_MIN,
        tau=0.001,
        max_size=100000,
        batch_size=64,
        update_rate=4,
        n_states=0,
        n_actions=0,
    ):

        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.tau = tau
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.action_space = [i for i in range(n_actions)]

        # Replay memory
        self.memory = ReplayMemory(max_size=max_size, n_states=n_states)

        # Q-Network
        self.qnetwork_local = QNetwork(lr=alpha, n_states=n_states, n_actions=n_actions)
        self.qnetwork_target = QNetwork(lr=alpha, n_states=n_states, n_actions=n_actions)

        self.counter = 0

    def decrement_epsilon(self):
        self.eps *= self.eps_decay
        if self.eps < self.eps_min:
            self.eps = self.eps_min

    def epsilon_greedy(self, state):
        if np.random.random() > self.eps:
            state = T.tensor([state], dtype=T.float).to(
                self.qnetwork_local.device)
            actions = self.qnetwork_local.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        states, actions, rewards, next_states, dones = \
            self.memory.sample_memory(self.batch_size)
        t_states = T.tensor(states).to(self.qnetwork_local.device)
        t_actions = T.tensor(actions).to(self.qnetwork_local.device)
        t_rewards = T.tensor(rewards).to(self.qnetwork_local.device)
        t_next_states = T.tensor(next_states).to(self.qnetwork_local.device)
        t_dones = T.tensor(dones).to(self.qnetwork_local.device)
        return t_states, t_actions, t_rewards, t_next_states, t_dones

    def save_models(self):
        self.qnetwork_local.save_checkpoint()
        self.qnetwork_target.save_checkpoint()

    def load_models(self):
        self.qnetwork_local.load_checkpoint()
        self.qnetwork_target.load_checkpoint()

    def learn(self, state, action, reward, next_state, done):
        # Save experience to memory
        self.store_transition(state, action, reward, next_state, done)

        # If not enough memory then skip learning
        if self.memory.memory_counter < self.batch_size:
            return

        # Update target network parameter every update rate
        if self.counter % self.update_rate == 0:
            self.soft_update(self.tau)

        # Take random sampling from memory
        states, actions, rewards, next_states, dones = self.sample_memory()

        # Update action value
        indices = np.arange(self.batch_size)
        q_pred = self.qnetwork_local.forward(states)[indices, actions]
        q_next = self.qnetwork_target.forward(next_states).max(dim=1)[0]
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        self.qnetwork_local.optimizer.zero_grad()
        loss = \
            self.qnetwork_local.loss(q_target, q_pred) \
            .to(self.qnetwork_local.device)
        loss.backward()
        self.qnetwork_local.optimizer.step()
        self.counter += 1

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def regular_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data + target_param.data)


class DDQNAgent(object):
    def __init__(
            self,
            alpha=0.0005,
            gamma=0.99,
            eps=EPS,
            eps_decay=EPS_DECAY,
            eps_min=EPS_MIN,
            tau=0.001,
            max_size=100000,
            batch_size=64,
            update_rate=4,
            n_states=0,
            n_actions=0,
    ):

        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.tau = tau
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.action_space = [i for i in range(n_actions)]

        # Replay memory
        self.memory = ReplayMemory(max_size=max_size, n_states=n_states)

        # Q-Network
        self.qnetwork_local = QNetwork(lr=alpha, n_states=n_states, n_actions=n_actions)
        self.qnetwork_target = QNetwork(lr=alpha, n_states=n_states, n_actions=n_actions)

        self.counter = 0

    def decrement_epsilon(self):
        self.eps *= self.eps_decay
        if self.eps < self.eps_min:
            self.eps = self.eps_min

    def epsilon_greedy(self, state):
        if np.random.random() > self.eps:
            state = T.tensor([state], dtype=T.float).to(
                self.qnetwork_local.device)
            actions = self.qnetwork_local.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        states, actions, rewards, next_states, dones = \
            self.memory.sample_memory(self.batch_size)
        t_states = T.tensor(states).to(self.qnetwork_local.device)
        t_actions = T.tensor(actions).to(self.qnetwork_local.device)
        t_rewards = T.tensor(rewards).to(self.qnetwork_local.device)
        t_next_states = T.tensor(next_states).to(self.qnetwork_local.device)
        t_dones = T.tensor(dones).to(self.qnetwork_local.device)
        return t_states, t_actions, t_rewards, t_next_states, t_dones

    def save_models(self):
        self.qnetwork_local.save_checkpoint()
        self.qnetwork_target.save_checkpoint()

    def load_models(self):
        self.qnetwork_local.load_checkpoint()
        self.qnetwork_target.load_checkpoint()

    def learn(self, state, action, reward, next_state, done):
        # Save experience to memory
        self.store_transition(state, action, reward, next_state, done)

        # If not enough memory then skip learning
        if self.memory.memory_counter < self.batch_size:
            return

        # Update target network parameter every update rate
        if self.counter % self.update_rate == 0:
            self.soft_update(self.tau)

        self.qnetwork_local.optimizer.zero_grad()  # clear the gradient before back propragation

        # Take random sampling from memory
        states, actions, rewards, next_states, dones = self.sample_memory()

        # Update action value
        batch_index = np.arange(self.batch_size)

        # states = T.tensor(states, dtype= T.float) #.to(self.model.device) # turn np.array to pytorch tensor
        states_ = next_states  # T.tensor(, dtype= T.float) #.to(self.model.device)
        # rewards = T.tensor(rewards) #.to(self.model.device) # tensor([batchsize])
        terminals = T.tensor(dones, dtype=T.float)  # .to(self.model.device)

        '''Perform feedforward to compare: estimate value of current state (state) toward the max value of next state(states_)'''
        # We want the delta between action the agent actually took and max action
        # batch index loop trhough all state
        q_prediction = self.qnetwork_local(states)
        actions = T.tensor(actions, dtype=T.int64)  # .to(self.qnetwork_local.device)  # dont need to be a tensor

        q_s_a = q_prediction.gather(1, actions.unsqueeze(1)).squeeze()
        q_tp1_values = self.qnetwork_local(states_).detach()
        _, a_prime = q_tp1_values.max(1)

        # Get Q values from frozen network for next state and chosen action
        # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        q_target_tp1_values = self.qnetwork_target(states_).detach()
        q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
        q_target_s_a_prime = q_target_s_a_prime.squeeze()

        # If current state is end of episode, then there is no next Q value
        q_target_s_a_prime = (1 - terminals) * q_target_s_a_prime
        q_target = rewards + self.gamma * q_target_s_a_prime

        # Clip the error and flip
        loss = self.qnetwork_local.loss(q_s_a, q_target)
        loss.backward()  # back-propagate
        self.qnetwork_local.optimizer.step()  # update model weights
        self.counter += 1

    def soft_update(self, tau):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def regular_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(local_param.data + target_param.data)


class DeltaEnv:
    def __init__(self):

        self.min_theta = -1.1
        self.max_theta = 1.1
        self.theta = np.array([0.0, 0.0, 0.0])

        self.distance = 100
        self.current_distance = 100

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
        self.goal_pos = self.generate_random_positions()[1]
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
        info = {
            'distance': self.distance,
            'goal_position': self.goal_pos,
            'ee_position': self.ee_pos
        }
        #self.render(self.theta, self.goal_pos)
        # self.timerate.sleep()
        return observation, reward, done, info

    def generate_random_positions(self):
        angles = np.arange(-1.1, 1.1, 0.01).tolist()

        theta0 = random.choice(angles)
        theta1 = random.choice(angles)
        theta2 = random.choice(angles)

        self.goal = self.forward_kinematics(theta0, theta1, theta2)[0]

        return self.goal

    @staticmethod
    def normalize_angle(angle):
        return mt.atan2(mt.sin(angle), mt.cos(angle))

    def reset(self):
        self.goal_pos = self.generate_random_positions()

        self.theta[0] = 0
        self.theta[1] = 0
        self.theta[2] = 0

        self.current_score = 0

        self.ee_pos = self.forward_kinematics(
            self.theta[0], self.theta[1], self.theta[2])[0]
        observation = np.hstack(
            (self.goal_pos, self.ee_pos, self.theta, self.distance))

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


def train(algorithm='DQN'):
    env = DeltaEnv()
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # ou_noise = OUNoise(dim=1, low=0, high=14)
    # ou_noise.reset()

    if algorithm == 'DQN':
        agent = DQNAgent(alpha=0.0001, n_states=n_states, n_actions=n_actions, eps=0.2, eps_min=0.01)
    else:
        agent = DDQNAgent(alpha=0.0001, n_states=n_states, n_actions=n_actions, eps=0.2, eps_min=0.01)
    # writer = SummaryWriter(f'./log/{LOGS_DIR}')
    load_models = False
    n_episodes = 500
    n_steps = 500

    # Load weights
    if load_models:
        agent.eps = agent.eps_min
        agent.load_models()

    total_reward_hist = []
    avg_reward_hist = []
    num_steps = 0
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        for t in range(n_steps):
            num_steps += 1
            # Render after episode 1800
            #if episode > EPISODE_TO_START_PRINTING:
            #    env.render(env.theta, env.goal_pos)
            action = agent.epsilon_greedy(state)
            #print('Greedy:', type(action))
            # print('Noise:', int(ou_noise.get_action(action, num_steps).item()))
            # action = int(ou_noise.get_action(action, num_steps).item())
            next_state, reward, done, info = env.step(action)

            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        if not load_models:
            agent.decrement_epsilon()

        total_reward_hist.append(total_reward)
        avg_reward = np.average(total_reward_hist[-100:])
        avg_reward_hist.append(avg_reward)
        print("Episode :", episode, "Epsilon : {:.4f}".format(agent.eps), "Total Reward : {:.4f}".format(
            total_reward), "Avg Reward : {:.4f}".format(avg_reward))
    return total_reward_hist, avg_reward_hist, agent.qnetwork_local


total_reward_hist_dqn, avg_reward_hist_dqn, actor = train('DQN')
np.save('DQN_data.npy', avg_reward_hist_dqn)
T.save('DQN_agent.pt', actor)

total_reward_hist_ddqn, avg_reward_hist_ddqn, actor = train('DDQN')
np.save('DDQN_data.npy', avg_reward_hist_ddqn)
T.save('DDQN_agent.pt', actor)
