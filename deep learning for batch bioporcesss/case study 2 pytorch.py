import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Actor network (policy)
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # Outputs between -1 and 1

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

# Critic network (Q-value function)
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# Dynamics function
def dynamics(t, y, u):
    y1, y2 = y
    u1, u2 = u
    dy1dt = -(u1 + 0.5 * u1**2) * y1 + 0.5 * u2 * y2 / (y1 + y2)
    dy2dt = u1 * y1 - 0.7 * u2 * y1
    return torch.tensor([dy1dt, dy2dt])
def reward_function(y2):
    return y2  # Simple reward: maximize y2


# ODE function for the system
def ode_func(t, y):
    with torch.no_grad():
        u = actor(y)  # Get actions (u1, u2) from the actor network
    return dynamics(t, y, u)

# Hyperparameters
state_size = 2
action_size = 2
hidden_size = 128
lr_actor = 0.001
lr_critic = 0.002
gamma = 0.99
tau = 0.005
buffer_size = 100000
batch_size = 64

# Instantiate networks
actor = Actor(state_size, hidden_size, action_size)
actor_target = Actor(state_size, hidden_size, action_size)
critic = Critic(state_size, action_size, hidden_size)
critic_target = Critic(state_size, action_size, hidden_size)

# Copy initial weights from actor to actor target, and critic to critic target
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

# Replay buffer
replay_buffer = ReplayBuffer(buffer_size, batch_size)

# Soft update of target network parameters
def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
# Training loop
n_episodes = 500
reward_list = []

for episode in range(n_episodes):
    state = torch.tensor([1.0, 0.0], dtype=torch.float32)  # Initial state [y1, y2]
    t = torch.linspace(0., 1., 100)
    
    # Solve ODE using actor policy
    with torch.no_grad():
        sol = odeint(ode_func, state, t, method='rk4', options={'step_size': 0.01})
    
    # Convert solution to usable form
    y1_sol = sol[:, 0].detach().numpy()
    y2_sol = sol[:, 1].detach().numpy()
    
    total_reward = 0
    for i in range(len(t)):
        next_state = sol[i].detach().numpy()
        done = (i == len(t) - 1)
        
        reward = reward_function(next_state[1])  # Using y2 for reward
        total_reward += reward

        # Store experience in replay buffer
        replay_buffer.add((state.numpy(), actor(state).detach().numpy(), reward, next_state, done))
        state = torch.tensor(next_state, dtype=torch.float32)

        # Training step
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample()
            
            # Critic update
            next_actions = actor_target(next_states)
            Q_targets_next = critic_target(next_states, next_actions)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = critic(states, actions)
            critic_loss = nn.MSELoss()(Q_expected, Q_targets.detach())
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # Actor update
            actions_pred = actor(states)
            actor_loss = -critic(states, actions_pred).mean()
            
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            # Soft update target networks
            soft_update(critic, critic_target, tau)
            soft_update(actor, actor_target, tau)
    
    reward_list.append(total_reward)
    print(f"Episode {episode+1}/{n_episodes}, Total Reward: {total_reward}")

# Plot rewards over episodes
plt.plot(reward_list)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DDPG Training Progress')
plt.show()

# Plot final states y1 and y2 after training
y1_sol = sol[:, 0].detach().numpy()
y2_sol = sol[:, 1].detach().numpy()

plt.plot(t, y1_sol, label=r'$y_1$')
plt.plot(t, y2_sol, label=r'$y_2$')
plt.xlabel('Time')
plt.ylabel(r'$y$')
plt.legend()
plt.title('Final States After Training')
plt.show()