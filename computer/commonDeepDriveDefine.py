#this file is here to define all important constant that could be used everywhere


#CAR_IP = '192.168.1.5'    # Server(Raspberry Pi) IP address
#CAR_IP = '192.168.1.6'    # Server(Raspberry Pi) IP address
#CAR_IP = '10.246.50.143'    # Server(Raspberry Pi) IP address
CAR_IP = '10.246.51.13'    # Server(Raspberry Pi) IP address

PORT_VIDEO_SERVER = 8000
PORT_STEER_SERVER = 8001
PORT_SENSOR_SERVER = 8002
ADDR_VIDEO_SERVER = (CAR_IP, PORT_VIDEO_SERVER)
ADDR_STEER_SERVER = (CAR_IP, PORT_STEER_SERVER)
ADDR_SENSOR_SERVER = (CAR_IP, PORT_SENSOR_SERVER)

#client to enable
videoClientEnable = True
steerClientEnable = True
sensorClientEnable = False


# min max angle for the car
MIN_ANGLE    = -50
MAX_ANGLE    = 50

# STEP_CAPTURE should always be lower than STEP_REPLAY
STEP_CAPTURE = 1
#step replay define the number of step during replay or autonomous drive
#it is used to define the number of neural network as well
STEP_REPLAY  = 25


# Number of NeuralNetwork output =
#  => (MAX - MIN) / STEP : Number of values except 0
# + 1 to take into account boundary or 0
NN_OUTPUT_NUMBER = (MAX_ANGLE - MIN_ANGLE) / STEP_REPLAY + 1

# value of video FPS used to determine for instance runnin gaverage
VIDEO_FPS = 30

#sampling of the image into vstack array (FPS record)
FPS_RECORD_TIME = 0.1

# sampling time for
STEERING_KEYBOARD_SAMPLING_TIME = 0.04

# this is the max delta angle allowed between prediction and keyboard control
MAX_KEYBOARD_DELTA_ANGLE_TO_PREDICTION = 15

#Steering sampling time for prediction. it used for prediction running average table and update steer when prediction is good
STEERING_PREDICTION_SAMPLING_TIME = 0.1

#number of sample for running average
NB_SAMPLE_RUNNING_AVERAGE_PREDICTION = int(VIDEO_FPS * STEERING_PREDICTION_SAMPLING_TIME) + 1

# Image size
ROWS = 120
COLS = 320

####### class dedicated to all thread/socket control

class ClientCommand(object):
    """ A command to the client thread.
        Each command type has its associated data:

        CONNECT:    (host, port) tuple
        SEND:       Data string
        RECEIVE:    None
        CLOSE:      None
    """
    CONNECT, CONNECT_STREAM, SEND, RECEIVE, RECEIVE_IMAGE, STOP, CLOSE  = range(7)

    def __init__(self, type, data=None):
        self.type = type
        self.data = data


class ClientReply(object):
    """ A reply from the client thread.
        Each reply type has its associated data:

        ERROR:      The error string
        SUCCESS:    Depends on the command - for RECEIVE it's the received
                    data string, for others None.
    """
    ERROR, SUCCESS = range(2)

    def __init__(self, type, data=None):
        self.type = type
        self.data = data
