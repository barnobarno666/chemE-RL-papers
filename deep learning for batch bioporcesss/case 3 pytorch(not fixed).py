import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# Define the system of ODEs with parameters as arguments
def system(t, y, um=0.5, ks=0.1, ki=0.5, KN=10.0, YN_X=1.0, km=0.5, kd=0.1):
    # Unpack concentrations
    cx, cN, cq = y
    
    # ODE equations
    d_cx_dt = (um * (t / (t + ks)) * (cx / (cN + KN))) - (0.1 * cx)  # Example biomass growth model
    d_cN_dt = -YN_X * (um * (t / (t + ks)) * (cx / (cN + KN))) + 1.0  # Nitrate consumption
    d_cq_dt = (km * (t / (t + ks)) * (cx / (cN + KN)) - kd * (cq / (cN + KN))) if (cN <= 500 and cx >= 10) else 0

    return torch.tensor([d_cx_dt, d_cN_dt, d_cq_dt])

# Define initial conditions
initial_conditions = torch.tensor([1.0, 150.0, 0.0])  # [cx(0), cN(0), cq(0)]
time_points = torch.linspace(0, 100, 100)  # Time from 0 to 100 seconds

# Integrate the system
def integrate_system(system, initial_conditions, time_points):
    # Solve ODE
    solution = odeint(system, initial_conditions, time_points)
    return solution

# Disturbance parameters
sigma_d = torch.tensor([4e-3, 1.0, 1e-7])

# Measurement noise parameters
sigma_n = torch.tensor([4e-4, 0.1, 1e-8])

# Define disturbance and noise functions
def add_disturbance(t, sigma_d):
    """Additive disturbance function."""
    disturbance = (torch.sin(t) * sigma_d[0], sigma_d[1], sigma_d[2])
    return torch.tensor(disturbance)

def add_noise(sigma_n):
    """Measurement noise function."""
    noise = torch.normal(0, sigma_n)
    return noise

# Define the reward function
def calculate_reward(solution, time_points):
    """Calculate the reward based on the change of control actions and final product concentration."""
    final_cq = solution[-1, 2]  # Product concentration at the final time point
    change_in_actions = (solution[-1, 1] - solution[0, 1]) ** 2  # Change in control actions (simplified)
    reward = -change_in_actions * 3.125e-6 + final_cq  # Incorporating control action change and final product concentration
    return reward.item()  # Convert to a Python float for convenience

# Integrate the dynamic system
solution = integrate_system(system, initial_conditions, time_points)

# Simulating disturbance and noise during integration
disturbances = [add_disturbance(t, sigma_d) for t in time_points]
noises = [add_noise(sigma_n) for _ in time_points]

# Adjust the solution with disturbances and noise
adjusted_solution = solution + torch.stack(disturbances) + torch.stack(noises)

# Calculate the reward based on the adjusted solution
reward = calculate_reward(adjusted_solution, time_points)

# Print the calculated reward
print(f'Calculated Reward: {reward:.4f}')

# Plotting the results
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time_points.numpy(), adjusted_solution[:, 0].numpy(), label='Biomass Concentration (X)')
plt.title('Concentration Over Time')
plt.ylabel('Biomass (g/L)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_points.numpy(), adjusted_solution[:, 1].numpy(), label='Nitrate Concentration (N)', color='orange')
plt.ylabel('Nitrate (mg/L)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_points.numpy(), adjusted_solution[:, 2].numpy(), label='Product Concentration (q)', color='green')
plt.ylabel('Product Concentration (q)')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.show()
