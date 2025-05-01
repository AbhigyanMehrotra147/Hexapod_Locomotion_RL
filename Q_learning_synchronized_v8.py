import my_hexapod_synchronized_legs_env as rvd
import numpy as np
import pybullet as p
from time import sleep
import pybullet_data
import os
import matplotlib.pyplot as plt


"""
The goal of this Q learning is to just walk and not much else. 
"""

client = p.connect(p.DIRECT)
p.setTimeStep(rvd.time_step / rvd.frameskip)
p.resetSimulation()
p.setGravity(0, 0, -9.81, physicsClientId = client)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.stepSimulation()
p.setRealTimeSimulation(0)

# Q learning parameters
alpha = 0.1
gamma = 0.95
epsilon = 1
epsilon_decay = 0.999
min_epsilon = 0.01
episodes = 7000

# The Q table 
Q_table = np.zeros((rvd.Total_states,rvd.Total_actions))


Hexapod = p.loadURDF("Hexapod.urdf", basePosition = [0, 0, 0.2])
Plane = p.loadURDF("Plane.urdf")
joint_list = [j for j in range(p.getNumJoints(Hexapod)) if p.getJointInfo(Hexapod, j)[2] == p.JOINT_REVOLUTE]
joints_by_leg = [joint_list[i:i+3] for i in range(0, len(joint_list), 3)]





total_rewards_list = []
num_steps_walked_per_episode = []
for episode in range(episodes):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    Hexapod = p.loadURDF("Hexapod.urdf", basePosition=rvd.INIT_POSITION)
    Plane = p.loadURDF("Plane.urdf")

    rvd.observations = rvd.update_observation(p=p,Hexapod=Hexapod)
    state_index = rvd.get_state_index(rvd.observations)

    done = False
    total_reward = 0
    count = 0
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(rvd.Total_actions)
        else:
            action = np.argmax(Q_table[state_index])

        # Apply action
        transformed_pos = rvd.get_target_positions(action)
        max_torques = [rvd.servo_max_torque] * rvd.num_motors
        

        p.setJointMotorControlArray(bodyIndex = Hexapod, jointIndices = joint_list, controlMode = p.POSITION_CONTROL, targetPositions = transformed_pos, forces = max_torques)
        for _ in range(rvd.frameskip):
            p.stepSimulation()
            sleep(rvd.dt / rvd.frameskip)
        
        # Get next observation and reward
        rvd.observations = rvd.update_observation(p=p,Hexapod=Hexapod)
        next_state_index = rvd.get_state_index(rvd.observations)
        reward = rvd.get_reward(20,1,0.1,p,Hexapod)

        # Q-value update
        Q_table[state_index, action] += alpha * (
            reward + gamma * np.max(Q_table[next_state_index]) - Q_table[state_index, action]
        )



        state_index = next_state_index
        total_reward += reward

        position, _ = p.getBasePositionAndOrientation(Hexapod)
        # If the robot has fallen
        done = bool(position[2] < 0.08)
        count += 1



    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if (episode % 50 == 0):
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Count: {count}")

        num_steps_walked_per_episode.append(count)
        total_rewards_list.append(total_reward)


plt.plot(total_rewards_list,np.arange(len(total_rewards_list))*50)
plt.title("total_reward vs Episodes synchronized_Q_learning v8")
plt.xlabel("Num Episodes")
plt.ylabel("Total Reward")
plt.show()
plt.savefig("total_reward vs Episodes synchronized_Q_learning v8.png")

plt.plot(num_steps_walked_per_episode,np.arange(len(num_steps_walked_per_episode))*50)
plt.title("steps vs Episodes synchronized Q learning v8.")
plt.xlabel("Num Episodes")
plt.ylabel("Steps")
plt.show()
plt.savefig("steps vs Episodes synchronized Q learning v8.png")


np.save("Q_table_synchronized v8.npy", Q_table)


