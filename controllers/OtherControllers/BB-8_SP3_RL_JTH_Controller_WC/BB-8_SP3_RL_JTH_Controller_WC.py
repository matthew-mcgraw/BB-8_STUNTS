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

        self.check_point_list_min = list(np.zeros(4))
        self.check_point_list_max = list(np.ones(4))
        self.check_point_list = list(np.zeros(4))
        
        
        #self.observation_space = Box(low=np.array([-4, -6,-1, -1, -1, -math.pi,0,0,0,0]+self.check_point_list_min),
        #                             high=np.array([4, 6, 1, 1, 1, math.pi,8,8,12,12]+self.check_point_list_max),
        #                             dtype=np.float64)
                                     #x,y,z,xAng,yAng,zAng,Angle,twDist,bwDist,lwDist,rwDist,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z
        self.observation_space = Box(low=np.array([-5, -1.5,-0.2,-1, -1, -1, -math.pi,0,0,0,0,-4,-4,-4,-15,-15,-15]),
                                     high=np.array([5, 1.5,1, 1, 1, 1, math.pi,8,8,12,12,4,4,4,15,15,15]),
                                     dtype=np.float64)
        # Define agent's action space using Gym's Discrete
        #self.action_space = Discrete(2)
        
        #For BB-8, it will be increase/decrease body pitch motor speed, increase/decrease body yaw motor speed
        #OR ALL MOTORS OFF
        self.action_space = Discrete(5)

        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods

        self.robot_node = self.getFromDef("BB-8")

        self.body_yaw_motor = self.getDevice("body yaw motor")
        self.body_yaw_motor.setPosition(float("inf"))
        self.body_yaw_motor.setVelocity(0.0)
        self.yaw_speed = 0.0

        self.body_pitch_motor = self.getDevice("body pitch motor")
        self.body_pitch_motor.setPosition(float("inf"))
        self.body_pitch_motor.setVelocity(0.0)
        self.pitch_speed = 0.0
        
        self.body_acc = self.getDevice("body accelerometer")
        self.body_acc.enable(150)
        print(self.body_acc.getValues())

        self.body_gyro = self.getDevice("body gyro")
        self.body_gyro.enable(150)
        print(self.body_gyro.getValues())
        
        
        self.steps_per_episode = 1000  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
        self.YAW_MAX_SPEED = 8.72/4
        self.PITCH_MAX_SPEED = 8.72/2
        self.YAW_ATTENUATION = self.YAW_MAX_SPEED
        self.PITCH_ATTENUATION = self.PITCH_MAX_SPEED 
        
    def JUMP(self):
    
        x,y,z = self.get_translation()
        if z < 0:
            self.robot_node.addForce([0,0,3000],False)
            
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
        #print(self.body_acc.getValues())
        x, y, z = self.get_translation()
        x = round(x,1)
        y = round(y,1)
        z = round(z,3)
        #print(x)
        x = normalize_to_range(x, -5, 5, -1.0, 1.0)
        y = normalize_to_range(y, -1.5, 1.5, -1.0, 1.0)
        z = normalize_to_range(z,-0.2,1,-1.0,1.0)
        tw_dist = normalize_to_range(5-x,0,10,-1.0,1.0)
        bw_dist = normalize_to_range(x-(-5),0,10,-1.0,1.0)
        lw_dist = normalize_to_range(1.5-y,0,3,-1.0,1.0)
        rw_dist = normalize_to_range(y-(-1.5),0,3,-1.0,1.0)
        
        #print(x, y)
        
        xAng, yAng, zAng, Angle = self.get_rotation()
        xAng = round(xAng,2)
        yAng = round(yAng,2)
        zAng = round(zAng,2)
        Angle = round(Angle*20)/20
        #print(xAng,yAng,zAng,Angle)
        Angle = normalize_to_range(Angle, -math.pi, math.pi, -1.0, 1.0)
        self.step_counter += 1
        
        
        gx,gy,gz = self.body_gyro.getValues()
        gx = normalize_to_range(gx,-4,4,-1.0,1.0)
        gy = normalize_to_range(gy,-4,4,-1.0,1.0)
        gz = normalize_to_range(gz,-4,4,-1.0,1.0)
        
        ax,ay,az = self.body_acc.getValues()
        ax = normalize_to_range(ax,-15,15,-1.0,1.0)
        ay = normalize_to_range(ay,-15,15,-1.0,1.0)
        az = normalize_to_range(az,-15,15,-1.0,1.0)
        
        #print(self.intersecting_obstacle_plane())
        #print(self.intersecting_goal_plane())
        #chkPts = self.check_for_checkpoints()
        obs = [x,y,z,xAng,yAng,zAng,Angle,tw_dist,bw_dist,lw_dist,rw_dist,gx,gy,gz,ax,ay,az]
        return obs

    def get_default_observation(self):
            
        self.step_counter = 1
        self.past_obstacle = False
        x, y, z = self.get_translation()
        self.prev_x = x
        self.prev_y = y
        self.check_point_list = list(np.zeros(4))
        self.completed_lap = False
        #print(self.starting_x, self.starting_y)
        return [0.0 for _ in range(self.observation_space.shape[0])]

    def get_reward(self, action=None):
        # Reward is +1 for every step the episode hasn't ended
        #return 1
        reward = 0
        x, y, z = self.get_translation()
        
        return  - (4-x)/100
        
    def is_done(self):
        
        #if self.episode_score < -500:
        #    return True
        x,y,z = self.get_translation()
        #if x <-7.25 or x > 7.25 or y < -9.75 or y > 9.75:
            #self.episode_score -= 101
           #self.episode_score -= 125
        #    self.episode_score += normalize_to_range(-10,-100,100,-1,1) 
        #    return True
        if x > 1:
            self.episode_score += 100
            return True
            
        
        return False

    def solved(self):
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            print(np.mean(self.episode_score_list[-100:]))
            if np.mean(self.episode_score_list[-100:]) > 90:  # Last 100 episodes' scores average value
                print(np.sum(self.episode_score_list[-100:]))
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
            self.yaw_speed = 0.0
        elif action == 2:
            self.yaw_speed = self.YAW_MAX_SPEED
            self.pitch_speed = 0.0
        elif action == 3:
            self.yaw_speed = -self.YAW_MAX_SPEED
            self.pitch_speed = 0.0
        else:
            self.JUMP()
            
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
                    use_cuda=True, batch_size=32, actor_lr = 0.0001, critic_lr = 0.00015)
#agent.load("C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\SP3_Agents\\sp2_JTH_01")
#for param in agent.actor_net.parameters():
#    param.requires_grad = False
#for param in agent.critic_net.parameters():
#    param.requires_grad = False#

#for param in agent.actor_net.parameters():
#    param.requires_grad = True
#for param in agent.critic_net.parameters():
#    param.requires_grad = True

solved = False
episode_count = 0
episode_limit = 100000
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and episode_count < episode_limit:
    observation = env.reset()  # Reset robot and get starting observation
    env.episode_score = 0
    if episode_count % 1000 == 0:    
        #save agent every thousand episodes
        agent.save("C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\SP3_Agents\\sp2_JTH_02")

    for step in range(env.steps_per_episode):
        # In training mode the agent samples from the probability distribution, naturally implementing exploration
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        # Step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # Save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)
        #print(step)

        if done:
            # Save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)
            solved = env.solved()  # Check whether the task is solved
            break

        env.episode_score += reward  # Accumulate episode reward
        observation = new_observation  # observation for next step is current step's new_observation
    agent.log_episode_score(env.episode_score,episode_count)
    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # Increment episode counter

agent.save("C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\SP3_Agents\\sp2_JTH_02")
if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()