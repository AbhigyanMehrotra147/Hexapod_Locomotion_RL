import math
import numpy as np
from gym import spaces
from time import sleep

# Setting Action Constraints: 
action_upper_bound = 1.5
action_lower_bound = -1.5

time_step = 0.05
frameskip = 12

dt = time_step


"""
File to store variable definations and functionsn for discritizing states and actions
"""
num_motors = 18
num_legs = 6

"""
Number of actions for each motor
from spaces.Box we get a function which when sampled will give the actions for of the motors
"""
Num_actions_per_motor = 2
action_pos_space = spaces.Box(low = -1.5, high = 1.5, shape\
                               = (num_motors,), dtype = "float32")


"""
The size of the observation space is 4 beacuse 
1) Height of the base
2) 3 values for the orientation of the base

"""

n_observations = 6
observation_space = spaces.Box(low = -1, high = 1, shape = (n_observations,), dtype = "float32")
#Now I am also discritizing the observation: the states can have a value from -1 to 1 since they are normalized. 
discrete_observations = 3
# initializing observations as zeros initially: 
observations = np.zeros(n_observations,dtype = float)



num_state_observations = 4
# synchronizing the legs. 
Total_actions = Num_actions_per_motor**(num_motors//3)
Total_states = discrete_observations**num_state_observations
Q_table_size = int(Total_states*Total_actions)





# Energy constraints:
max_velocity = 59*2*math.pi/60
max_torque = 1.50041745
servo_max_speed = max_velocity
servo_max_torque = max_torque



print("Observation states function: ",observation_space)
print("Total number of actions: ",Total_actions)
print("Total number of states: ", Total_states)
print("Size of Q_table: ",Q_table_size)






# Setting target Position and initial Position
INIT_POSITION = [0, 0, 0.2]
TARGET_POSITION = [0, 0, 0.1]

# Function to be used to discritize the continous actions that the action_pos_space.sample() gives
def discritize(continous_value: float, num: int, high:float, low: float):
    array = np.linspace(low,high,num = num)
    for i in range(len(array)):
        if continous_value <= array[i]:
            return i
    return len(array) - 1

def update_observation(p,Hexapod):
    pos, ori = p.getBasePositionAndOrientation(Hexapod)
    # only height is relevant
    observations = [pos[2]] + list(p.getEulerFromQuaternion(ori))

    for i in range(len(observations)):
        observations[i] = discritize(continous_value=observations[i], num=discrete_observations,high=1, low = -1)
    return observations

def get_reward(w1,w2,w3,p,Hexapod):
    position, orientation = p.getBasePositionAndOrientation(Hexapod)
    ori = p.getEulerFromQuaternion(orientation)
    reward = 0 + w2 * (position[2] - TARGET_POSITION[2])  - w3 * (abs(ori[0]) + abs(ori[1]))
    reward *= 0.1
    return reward

def get_state_index(observations: np.array):
    state_index = 0
    for i in range(num_state_observations):
        state_index += observations[i]*(discrete_observations ** ((num_state_observations- 1)- i))
    return state_index

# converts action index to executable positions
def decimal_action_to_executable(number, output_base):
    digits = []
    while number > 0 or len(digits) < num_motors:
        digits.append(number % output_base)
        number //= output_base

    action_choosing_array = np.linspace(action_lower_bound,action_upper_bound,num = Num_actions_per_motor)
    for i in range(len(digits)):
        digits[i] = action_choosing_array[digits[i]]
    return digits




def get_target_positions(action, ):
    # since the actions are joint positions my target positions are the actions themselves
    abstract_actions = decimal_action_to_executable(number=action,output_base=Num_actions_per_motor)
    transformed_pos = list(range(18))
    transformed_pos[0] = abstract_actions[0]
    transformed_pos[1] = abstract_actions[1]
    transformed_pos[2] = abstract_actions[2]
    transformed_pos[9] = abstract_actions[0]
    transformed_pos[10] = abstract_actions[1]
    transformed_pos[11] = abstract_actions[2]
    transformed_pos[12] = abstract_actions[0]
    transformed_pos[13] = abstract_actions[1]
    transformed_pos[14] = abstract_actions[2]
    transformed_pos[3] = abstract_actions[3]
    transformed_pos[4] = abstract_actions[4]
    transformed_pos[5] = abstract_actions[5]
    transformed_pos[6] = abstract_actions[3]
    transformed_pos[7] = abstract_actions[4]
    transformed_pos[8] = abstract_actions[5]
    transformed_pos[15] = abstract_actions[3]
    transformed_pos[16] = abstract_actions[4]
    transformed_pos[17] = abstract_actions[5]

    transformed_pos = np.array(transformed_pos) * math.pi/2
    # transformed_pos
    return transformed_pos
