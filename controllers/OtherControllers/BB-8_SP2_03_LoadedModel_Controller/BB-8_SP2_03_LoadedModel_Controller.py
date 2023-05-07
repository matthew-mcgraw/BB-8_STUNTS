from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent_modifed import PPOAgent, Transition

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
        self.observation_space = Box(low=np.array([-7.5, -10, 0,-1, -1, -1, -math.pi]),
                                     high=np.array([7.5, 10, 25, 1, 1, 1, math.pi]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        #self.action_space = Discrete(2)
        
        #For BB-8, it will be increase/decrease body pitch motor speed, increase/decrease body yaw motor speed
        #OR ALL MOTORS OFF
        self.action_space = Discrete(4)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods

        #self.position_sensor = self.getDevice("polePosSensor")
        #self.position_sensor.enable(self.timestep)

        #self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        
        self.robot_node = self.getFromDef("BB-8")
        self.obstacle_node = self.getFromDef("OBSTACLE")
        self.goal_node = self.getFromDef("GOAL")
        
        self.prev_x = 0
        self.prev_y = 0
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
        
        
        self.steps_per_episode = 2000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        self.YAW_MAX_SPEED = 8.72/4
        self.PITCH_MAX_SPEED = 8.72/2
        self.YAW_ATTENUATION = self.YAW_MAX_SPEED
        self.PITCH_ATTENUATION = self.PITCH_MAX_SPEED 
            
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
        
    def intersecting_obstacle_plane(self):
        obs_trans_field = self.obstacle_node.getField("translation")
        obs_trans_values = obs_trans_field.getSFVec3f()
        
        obs_upper_x = obs_trans_values[0] + 13/2 + 0.25
        obs_lower_x = obs_trans_values[0] - 13/2 - 0.25
        obs_upper_y = obs_trans_values[1] + 1/2 + 0.25
        obs_lower_y = obs_trans_values[1] - 1/2 - 0.25
        
        #BOUNDS OF THE OBSTACLE PLANE
        #print(obs_upper_x)
        #print(obs_lower_x)
        #print(obs_upper_y)
        #print(obs_lower_y)
        
        robot_trans_field = self.robot_node.getField("translation")
        robot_trans_values = robot_trans_field.getSFVec3f()
        robot_x = robot_trans_values[0]
        robot_y = robot_trans_values[1]
        
        #print(robot_x)
        #print(robot_y)
        
        if(robot_x >= obs_lower_x and robot_x <= obs_upper_x and robot_y >= obs_lower_y and robot_y <= obs_upper_y):
            return True
        else:
            return False
        
    def intersecting_goal_plane(self):
        goal_trans_field = self.goal_node.getField("translation")
        goal_trans_values = goal_trans_field.getSFVec3f()
        
        goal_upper_x = goal_trans_values[0] + 5.5/2 + 0.25
        goal_lower_x = goal_trans_values[0] - 5.5/2 - 0.25
        goal_upper_y = goal_trans_values[1] + 1/2 + 0.25
        goal_lower_y = goal_trans_values[1] - 1/2 - 0.25
        
        #BOUNDS OF THE OBSTACLE PLANE
        #print(obs_upper_x)
        #print(obs_lower_x)
        #print(obs_upper_y)
        #print(obs_lower_y)
        
        robot_trans_field = self.robot_node.getField("translation")
        robot_trans_values = robot_trans_field.getSFVec3f()
        robot_x = robot_trans_values[0]
        robot_y = robot_trans_values[1]
        
        #print(robot_x)
        #print(robot_y)
        
        if(robot_x >= goal_lower_x and robot_x <= goal_upper_x and robot_y >= goal_lower_y and robot_y <= goal_upper_y):
            return True
        else:
            return False

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
        
        goal_trans_field = self.goal_node.getField("translation")
        goal_trans_values = goal_trans_field.getSFVec3f()

        x = normalize_to_range(x, -7.5, 7.5, -1.0, 1.0)
        y = normalize_to_range(y, -10, 10, -1.0, 1.0)
        
        dist_to_goal = math.sqrt((x-goal_trans_values[0])**2 + (y-goal_trans_values[1])**2)
        dist_to_goal = normalize_to_range(dist_to_goal,0,25,-1.0,1.0)
        
        xAng, yAng, zAng, Angle = self.get_rotation()
        Angle = normalize_to_range(Angle, -math.pi, math.pi, -1.0, 1.0)
        
        #print(self.intersecting_obstacle_plane())
        #print(self.intersecting_goal_plane())
        

        return [x,y,dist_to_goal,xAng,yAng,zAng,Angle]

    def get_default_observation(self):
        # This method just returns a zero vector as a default observation
        x, y, z = self.get_translation()
        self.starting_x = x
        self.starting_y = y
        self.prev_x = x
        self.prev_y = y
        #print(self.starting_x, self.starting_y)
        return [0.0 for _ in range(self.observation_space.shape[0])]
    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        #return 1
        x, y, z = self.get_translation()
        #distance = math.sqrt(x ** 2 + y ** 2) - 49
        #time = super().getTime()
        #reward = distance/100 #/ time
        #print("REWARD: ", reward)        
        goal_trans_field = self.goal_node.getField("translation")
        goal_trans_values = goal_trans_field.getSFVec3f()
        dist_to_goal = math.sqrt((x-goal_trans_values[0])**2 + (y-goal_trans_values[1])**2)
        
        intersect_obstacle = self.intersecting_obstacle_plane()
        if intersect_obstacle == True:
            return 0#-10
        
        intersect_goal = self.intersecting_goal_plane()
        if intersect_goal == True:
            return 50
        else:
            #x, y, z = self.get_translation()
            #distance = math.sqrt((self.starting_x - x) ** 2 + (self.starting_y - y) **2) - 20
            #return distance / 100
            return 0#-dist_to_goal/1000

    def is_done(self):
        
        #if self.episode_score < -500:
        #    return True
            
        intersect_obstacle = self.intersecting_obstacle_plane()
        if intersect_obstacle == True:
            self.episode_score -= 0
            print("Hit the wall")
            return True
        
        intersect_goal = self.intersecting_goal_plane()
        if intersect_goal == True:
            self.episode_score += 50 
            print("Reached the goal")
            return True
        else:       
            return False

    def solved(self):
        if len(self.episode_score_list) > 200:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-200:]) > 18:  # Last 100 episodes' scores average value
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
            self.pitch_speed = self.PITCH_MAX_SPEED
        elif action == 2:
            self.yaw_speed = self.YAW_MAX_SPEED
        else:
            self.yaw_speed = -self.YAW_MAX_SPEED
            
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
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0], number_of_actor_outputs=env.action_space.n, 
                    use_cuda=True, batch_size=32, actor_lr = 0.001, critic_lr = 0.002)
agent.load("C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\SP2_Agents\\sp2_07")
#for param in agent.actor_net.parameters():
#    param.requires_grad = False
#for param in agent.critic_net.parameters():
#    param.requires_grad = False
#for param in agent.actor_net.parameters():
#    param.requires_grad = True
#for param in agent.critic_net.parameters():
#    param.requires_grad = True

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()