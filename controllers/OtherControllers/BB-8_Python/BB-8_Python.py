"""
Sample controller of the Sphero's BB-8 robot.
The auto-pilot feature is enabled.
You can pilot the robot by selecting it, and using the computer keyboard:
- 'up/down' arrow keys:    move the BB-8 forward or backwards.
- 'left/right' arrow keys: spin the BB-8 to the left or to the right.
- 'space' key:             stop the BB-8 motors.
- 'A' key:                 toggle on/off the auto-pilot mode.
"""

from controller import Camera, Keyboard, Motor, Robot, Supervisor

# constants
MAX_SPEED = 8.7
ATTENUATION = 8.7/10
YAW_ATTENUATION = 8.7/10
MAX_YAW_SPEED = 8.72

# initialize the robot
#obot = Robot()
# get the time step of the current world


supervisor = Supervisor()
robot = supervisor.getFromDef("BB-8")

robot = supervisor.getFromDef("BB-8")
print(robot)
if robot is None:
    sys.stderr.write("No DEF MY_ROBOT node found in the current world file\n")
    sys.exit(1)
time_step = int(robot.getBasicTimeStep())

# init the motors in control by velocity
body_yaw_motor = robot.getDevice("body yaw motor")
body_yaw_motor.setPosition(float("inf"))
body_yaw_motor.setVelocity(0.0)

body_pitch_motor = robot.getDevice("body pitch motor")
body_pitch_motor.setPosition(float("inf"))
body_pitch_motor.setVelocity(0.0)

head_yaw_motor = robot.getDevice("head yaw motor")
head_yaw_motor.setPosition(float("inf"))
head_yaw_motor.setVelocity(0.0)


body_acc = robot.getDevice("body accelerometer")
body_acc.enable(150)
print(body_acc.getValues())

body_gyro = robot.getDevice("body gyro")
body_gyro.enable(150)
print(body_gyro.getValues())





def getTranslation(robotNode):
    trans_field = robotNode.getField("translation")
    values = trans_field.getSFVec3f()
    print("MY_ROBOT is at position: %g %g %g" % (values[0], values[1], values[2]))
    return values[0], values[1], values[2]
    
def getRotation(robotNode):
    rot_field = robotNode.getField("rotation")
    values = rot_field.getSFVec3f()
    print("MY_ROBOT is at rotation: %g %g %g %g" % (values[0], values[1], values[2], values[3]))
    return values[0], values[1], values[2], values[3]
    
getRotation(bb8_node)

# enable the camera if it is present on the robot
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    if device.getNodeType() == Camera.node_type:
        camera = Camera(device)
        camera.enable(time_step)

# enable the computer keyboard
keyboard = Keyboard()
keyboard.enable(time_step)

# mode
class Mode:
    AUTOPILOT = 0
    MANUAL = 1

mode = Mode.AUTOPILOT

# speeds
yaw_speed = 0.0
pitch_speed = 0.0

# main loop
while robot.step(time_step) != -1:
    # manual mode
    key = keyboard.getKey()
    if key != -1:
        if mode == Mode.AUTOPILOT:
            yaw_speed = 0.0
            pitch_speed = 0.0
        mode = Mode.MANUAL

        if key == ord('A'):
            mode = Mode.AUTOPILOT
        elif key == Keyboard.UP:
            pitch_speed += ATTENUATION
        elif key == Keyboard.DOWN:
            pitch_speed -= ATTENUATION
        elif key == Keyboard.RIGHT:
            yaw_speed -= YAW_ATTENUATION
        elif key == Keyboard.LEFT:
            yaw_speed += YAW_ATTENUATION
        elif key == ord(' '):
            yaw_speed = 0.0
            pitch_speed = 0.0
    print(body_acc.getValues())
    print(body_gyro.getValues())
    print(body_acc.getLookupTable())
    # speed attenuation
    pitch_speed = min(MAX_SPEED, max(-MAX_SPEED, ATTENUATION * pitch_speed))
    yaw_speed = min(MAX_YAW_SPEED, max(-MAX_YAW_SPEED, YAW_ATTENUATION * yaw_speed))
    
    # autopilot mode
    if mode == Mode.AUTOPILOT:
        t = robot.getTime()
        if t > 1.0:
            yaw_speed = 0.0
            pitch_speed = MAX_SPEED

    # set the motor speeds
    body_yaw_motor.setVelocity(0)
    head_yaw_motor.setVelocity(0)
    body_pitch_motor.setVelocity(8.72/2)
    
    #print("YAW SPEED: {:.2f}".format(yaw_speed))
    getTranslation(bb8_node)
    getRotation(bb8_node)
    
# cleanup
robot.cleanup()
