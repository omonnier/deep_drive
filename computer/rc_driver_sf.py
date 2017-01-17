import cv2
import numpy as np
import math
import threading
import Queue
import time
from ClientSocket import *
from SteerSocket import *
from VideoSocket import *
from SensorSocket import *

# distance data measured by ultrasonic sensor
sensor_data = " "

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
sensorClientEnable = True


MIN_ANGLE    = -50
MAX_ANGLE    = 50
STEP_REPLAY  = 5

MAX_IMAGE_COUNT = 50

#steering time for update in s
STEERING_SAMPLING_TIME = 0.1


# Number of NeuralNetwork output = 
#  => (MAX - MIN) / STEP : Number of values except 0
#  =>  + 1 to handle angle = 0
#  =>  + 1 to handle stop command
number_output = (MAX_ANGLE - MIN_ANGLE) / STEP_REPLAY + 1

####################################### Neural Network definition ##############################


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ANN_MLP()

    def create(self):
        layer_size = np.int32([38400, 32, number_output])
        self.model.create(layer_size)
        self.model.load('mlp_xml/mlp.xml')

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


####################################### Deep Drive Thread ##############################

class DeepDriveThread(threading.Thread):
    
    def __init__(self):
        #call init
        threading.Thread.__init__(self)
        
        #create Video Stream Client Thread
        self.sctVideoStream = VideoThread()
        self.sctVideoStream.name = 'VideoSocketThread'
        self.sctVideoStream.start()
        

        #create Steer Client Thread
        self.sctSteer = SteerThread()
        self.sctSteer.name = 'SteerSocketThread'
        self.sctSteer.start()

        
        #create Sensor Client Thread
        self.sctSensor = SensorThread()
        self.sctSensor.name = 'SensorSocketThread'
        self.sctSensor.start()
        
        #connect All client
        self.ConnectClient()
             
        # create neural network
        self.model = NeuralNetwork()
        self.model.create()

        self.values_to_average  = np.zeros(MAX_IMAGE_COUNT, dtype=np.int)
        self.image_count = 0


    def ConnectClient(self):
        # loop until all client connected
        videoClientConnected = False
        steerClientConnected = False
        sensorClientConnected = False
        
        #launch connection thread for all client
        self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CONNECT, 'http://' + CAR_IP + ':' + str(PORT_VIDEO_SERVER) + '/?action=stream'))
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_STEER_SERVER))
        self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.CONNECT, ADDR_SENSOR_SERVER))

        while ( (videoClientConnected != videoClientEnable) or
                (steerClientConnected != steerClientEnable) or
                (sensorClientConnected != sensorClientEnable) ):

            #wait for .5 second before to check 
            time.sleep(0.5)
            
            if (videoClientConnected != videoClientEnable):
                try:
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        videoClientConnected=True
                        print 'Video stream server connected'
                except Queue.Empty:
                    print 'Video Client not connected'

            if (steerClientConnected != steerClientEnable):
                try:
                    reply = self.sctSteer.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        steerClientConnected=True
                        print 'Steer server connected'
                except Queue.Empty:
                    print 'Steer Client not connected'

            if (sensorClientConnected != sensorClientEnable):
                try:
                    reply = self.sctSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        sensorClientConnected=True
                        print 'Sensor server connected'
                except Queue.Empty:
                    print 'Sensor Client not connected'
            
        
    def run(self):
                
        #Send speed to car 
        print 'set Speed'
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',25)))
        #initial steer command set to stop
        try:
            print 'Start Main Thread for Deep Drive'
            
            #start receiver thread client to receive continuously data
            if videoClientEnable == True :
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE_IMAGE,''))

            if sensorClientEnable == True :
                 self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #start car to be able to see additioanl data
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))

            #init steer command to angle of 0 
            NNsteerCommand = 10

            lastSteerTime = time.time()
            while True:
                time0=time.time()
                
                ############################# Manage IMAGE for Deep neural network to extract Steer Command ###############
                try:
                    # try to see if image ready
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                
                        #check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
     
                        #print length as debug
                        #print 'length =' + str(len(self.sctVideoStream.lastImage))
                        
                        #decode jpg into array
                        image = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)

                        # lower half of the image
                        half_gray = image[120:240,:]

                        #cv2.imshow('image', image)
                        cv2.imshow('mlp_image', half_gray)

                        # reshape image
                        image_array = half_gray.reshape(1, 38400).astype(np.uint8)
                        
                        # Convert image in float 
                        image_array = np.asarray (image_array, np.float32)
                        #print image_array;

                        # neural network makes prediction
                        NNsteerCommand = self.model.predict(image_array)

                        #WARNING QUICK AND DIRTY FIX BEFORE WE UNDERSTAND WHY WE HAVE TOO MUCH 0 prediction
                        if NNsteerCommand != 0:
                            # Average prediction   
                            if self.image_count < MAX_IMAGE_COUNT:
                                self.values_to_average[self.image_count] = NNsteerCommand
                                self.image_count += 1
                            else:
                                print ' out of image array '


                   
                    else:
                        print 'Error getting image :' + str(reply.data)
                        break
                        
                except Queue.Empty:
                    #queue empty most of the time because image not ready
                    pass


                ############################# Get Sensor value ###############
                try:
                    # try to see if image ready
                    reply = self.sctSensor.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        if (reply.data < 50):
                            print 'sensor value = ' + str(reply.data)
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                        else:
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))
                        
                    else:
                        print 'Error getting Sensor :' + str(reply.data)
                        break
                        
                except Queue.Empty:
                    #queue empty most of the time because image not ready
                    pass


                ############### Control Car with all the input we can have ####################

                #get time for steer command. determine if we can send control or not
                timeNow = time.time()
                timeTarget = lastSteerTime + STEERING_SAMPLING_TIME
                if timeNow > timeTarget:
                    #print self.values_to_average[0:self.image_count]
                    if self.image_count > 0:
                        NNsteerCommand = np.sum (self.values_to_average[0:self.image_count], dtype=int) / self.image_count
                        print 'nbimage = ' + str(self.image_count) + ' , average_value = ' + str(NNsteerCommand)
                        self.image_count = 0
                    else:
                        print 'Warning ! No prediction found for the last image 
                    
                    #it s time to updat steer command
                    if NNsteerCommand != 0:
                        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('ANGLE_NN',NNsteerCommand)))
                    lastSteerTime = timeNow
                    print 'NNCommand = ',NNsteerCommand
                
        finally:
            print 'end rc driver'
            #stop and close all client and close them
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','home')))
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.STOP))
            self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
            #let 1 second for process to close
            time.sleep(1)
            #and make sure all of them ended properly
            self.sctVideoStream.join()
            self.sctSteer.join()
            self.sctSensor.join()
            print 'end finally'
            
if __name__ == '__main__':
    #create Deep drive thread and strt
    DDriveThread = DeepDriveThread()
    DDriveThread.name = 'DDriveThread'
    
    #start
    DDriveThread.start()

    DDriveThread.join()
    print 'end'


