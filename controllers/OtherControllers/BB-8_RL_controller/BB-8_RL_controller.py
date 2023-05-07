from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np
import math
import time

#from controller import Robot


class BB8Robot_GoFAST(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        # Define agent's observation space using Gym's Box, setting the lowest and highest possible values
        #self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
        #                             high=np.array([0.4, np.inf, 1.3, np.inf]),
        #                             dtype=np.float64)
        
        #For the BB8 robot in this environment, it will be x pos (-49, 49), y pos (-49, 49), MAYBE velocity
        #also let's use the rotation information
        # [xPos, yPos, xAng, yAng, zAng, Angle]
        self.observation_space = Box(low=np.array([-49, -49,-1, -1, -1, -math.pi]),
                                     high=np.array([49, 49, 1, 1, 1, math.pi]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        #self.action_space = Discrete(2)
        
        #For BB-8, it will be increase/decrease body pitch motor speed, increase/decrease body yaw motor speed
        #OR ALL MOTORS OFF
        self.action_space = Discrete(5)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods

        #self.position_sensor = self.getDevice("polePosSensor")
        #self.position_sensor.enable(self.timestep)

        #self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        
        self.robot_node = self.getFromDef("BB-8")
        #time.sleep(2.5)        
        #print(super().getTime())
        #get_translation()
        #print(self.robot_node)
        #trans_field = self.robot_node.getField("translation")
        #values = trans_field.getSFVec3f()
        #print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
        
        
        
        #self.wheels = []
        #for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
        #    wheel = self.getDevice(wheel_name)  # Get the wheel handle
        #    wheel.setPosition(float('inf'))  # Set starting position
        #    wheel.setVelocity(0.0)  # Zero out starting velocity
        #    self.wheels.append(wheel)
        
        self.body_yaw_motor = self.getDevice("body yaw motor")
        self.body_yaw_motor.setPosition(float("inf"))
        self.body_yaw_motor.setVelocity(0.0)
        self.yaw_speed = 0.0

        self.body_pitch_motor = self.getDevice("body pitch motor")
        self.body_pitch_motor.setPosition(float("inf"))
        self.body_pitch_motor.setVelocity(0.0)
        self.pitch_speed = 0.0
        
        
        self.steps_per_episode = 10000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        self.YAW_MAX_SPEED = 8.72
        self.PITCH_MAX_SPEED = 8.72
        self.YAW_ATTENUATION = self.YAW_MAX_SPEED/10
        self.PITCH_ATTENUATION = self.PITCH_MAX_SPEED/10    
            
    def get_translation(self):
        trans_field = self.robot_node.getField("translation")
        values = trans_field.getSFVec3f()
        #print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
        return values[0], values[1], values[2]
        
    def get_rotation(self):
        rot_field = self.robot_node.getField("rotation")
        values = rot_field.getSFVec3f()
        #print("MY_ROBOT is at rotation: %g %g %g %g" % (values[0], values[1], values[2], values[3]))
        return values[0], values[1], values[2], values[3]

    def get_observations(self):
        # Position on x-axis
        #cart_position = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # Linear velocity on x-axis
        #cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # Pole angle off vertical
        #pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # Angular velocity y of endpoint
        #endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)
        
        x, y, z = self.get_translation()
        x = normalize_to_range(x, -49, 49, -1.0, 1.0)
        y = normalize_to_range(y, -49, 49, -1.0, 1.0)
        
        xAng, yAng, zAng, Angle = self.get_rotation()
        Angle = normalize_to_range(Angle, -math.pi, math.pi, -1.0, 1.0)
        

        return [x,y,xAng,yAng,zAng,Angle]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        #return 1
        x, y, z = self.get_translation()
        distance = math.sqrt(x ** 2 + y ** 2) - 49
        time = super().getTime()
        reward = distance/100 #/ time
        #print("REWARD: ", reward)
        if reward is None:
            return 0
        else:
            return reward


    def is_done(self):
    
        x, y, z = self.get_translation()
        distance = math.sqrt(x ** 2 + y ** 2)
        if distance > 49:
            return True
            
        #time = super().getTime()
        #if time > 25:
        #    return True
            
        if self.episode_score < -1000:
            return True

        #pole_angle = round(self.position_sensor.getValue(), 2)
        #if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
        #    return True

        #cart_position = round(self.robot.getPosition()[0], 2)  # Position on x-axis
        #if abs(cart_position) > 0.39:
        #    return True

        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > -185:  # Last 100 episodes' scores average value
                return True
        return False

    def get_info(self):
        return None

    def render(self, mode='human'):
        pass

    def apply_action(self, action):
        action = int(action[0])
        #print(action)
        if action == 0:
            self.pitch_speed = 0.0
            self.yaw_speed = 0.0
        elif action == 1:
            self.pitch_speed += self.PITCH_ATTENUATION
        elif action == 2:
            self.pitch_speed -= self.PITCH_ATTENUATION
        elif action == 3:
            self.yaw_speed += self.YAW_ATTENUATION
        else:
            self.yaw_speed -= self.YAW_ATTENUATION
            
        self.pitch_speed = min(self.PITCH_MAX_SPEED, max(-self.PITCH_MAX_SPEED, self.pitch_speed))
        self.yaw_speed = min(self.YAW_MAX_SPEED, max(-self.YAW_MAX_SPEED, self.yaw_speed))
        
        #print(self.pitch_speed)
        #print("PITCH SPEED: ", self.pitch_speed)
        
        self.body_yaw_motor.setPosition(float("inf"))
        self.body_yaw_motor.setVelocity(self.yaw_speed)
        self.body_pitch_motor.setPosition(float("inf"))
        self.body_pitch_motor.setVelocity(self.pitch_speed)

        #for i in range(len(self.wheels)):
        #    self.wheels[i].setPosition(float('inf'))
        #    self.wheels[i].setVelocity(motor_speed)


env = BB8Robot_GoFAST()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n)

solved = False
episode_count = 0
episode_limit = 10000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation

    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")
    agent.save("C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\SP1_Agents\\sp1_01")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()