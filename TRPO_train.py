from torch.optim import Adam
from torch.distributions import Categorical
from collections import namedtuple
import torch
from torch import nn
import random
import numpy as np
from gym import spaces
import math as mt
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')


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


actor_hidden = 32
env = DeltaEnv()
state_size = env.observation_space.n
num_actions = env.action_space.n
device = 'cuda'

actor = nn.Sequential(nn.Linear(state_size, 300),
                      nn.ReLU(),
                      nn.Linear(300, 400),
                      nn.ReLU(),
                      nn.Linear(400, 600),
                      nn.ReLU(),
                      nn.Linear(600, 600),
                      nn.ReLU(),
                      nn.Linear(600, 400),
                      nn.ReLU(),
                      nn.Linear(400, 300),
                      nn.ReLU(),
                      nn.Linear(300, num_actions),
                      nn.Softmax(dim=1)).to(device)

Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', ])


def get_action(state):
    state = torch.tensor(state).float().unsqueeze(0).to(device)  # Turn state into a batch with a single element
    dist = Categorical(actor(state))  # Create a distribution from probabilities for actions
    return dist.sample().item()


# Critic takes a state and returns its values
#critic_hidden = 32

critic = nn.Sequential(nn.Linear(state_size, 300),
                      nn.ReLU(),
                      nn.Linear(300, 400),
                      nn.ReLU(),
                      nn.Linear(400, 600),
                      nn.ReLU(),
                      nn.Linear(600, 600),
                      nn.ReLU(),
                      nn.Linear(600, 400),
                      nn.ReLU(),
                      nn.Linear(400, 300),
                      nn.ReLU(),
                      nn.Linear(300, 1)).to(device)
#critic = nn.Sequential(nn.Linear(state_size, critic_hidden),
#                       nn.ReLU(),
#                       nn.Linear(critic_hidden, 1))
critic_optimizer = Adam(critic.parameters(), lr=0.005)


def update_critic(advantages):
    loss = .5 * (advantages ** 2).mean()  # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()


# delta, maximum KL divergence
max_d_kl = 0.01


def update_agent(rollouts):
    states = torch.cat([r.states for r in rollouts], dim=0).to(device)
    actions = torch.cat([r.actions for r in rollouts], dim=0).flatten().to(device)

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten().to(device)

    # Normalize advantages to reduce skewness and improve convergence
    advantages = (advantages - advantages.mean()) / advantages.std()

    update_critic(advantages)

    distribution = actor(states)
    distribution = torch.distributions.utils.clamp_probs(distribution)
    probabilities = distribution[range(distribution.shape[0]), actions]

    # Now we have all the data we need for the algorithm

    # We will calculate the gradient wrt to the new probabilities (surrogate function),
    # so second probabilities should be treated as a constant
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = kl_div(distribution, distribution)

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True)  # Create graph, because we will call backward() on it (for HVP)

    def HVP(v):
        return flat_grad(d_kl @ v, parameters, retain_graph=True)

    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            distribution_new = actor(states)
            distribution_new = torch.distributions.utils.clamp_probs(distribution_new)
            probabilities_new = distribution_new[range(distribution_new.shape[0]), actions]

            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = kl_div(distribution, distribution_new)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1


def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
    advantages = next_values - values
    return advantages


def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel


def trainTRPO(epochs=200, num_rollouts=500):
    mean_total_rewards = []
    global_rollout = 0

    for epoch in range(epochs):
        rollouts = []
        rollout_total_rewards = []

        for t in range(num_rollouts):
            state = env.reset()
            done = False

            samples = []

            while not done:

                with torch.no_grad():
                    action = get_action(state)

                next_state, reward, done, _ = env.step(action)

                # Collect samples
                samples.append((state, action, reward, next_state))

                state = next_state

            # Transpose our samples
            states, actions, rewards, next_states = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.as_tensor(actions).unsqueeze(1).to(device)
            rewards = torch.as_tensor(rewards).unsqueeze(1).to(device)

            rollouts.append(Rollout(states, actions, rewards, next_states))

            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1

        update_agent(rollouts)
        mtr = np.mean(rollout_total_rewards)
        print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')

        mean_total_rewards.append(mtr)
    return mean_total_rewards


mean_total_rewards = trainTRPO()
plt.plot(mean_total_rewards)
plt.show()
