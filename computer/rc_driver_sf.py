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
from datetime import datetime

# distance data measured by ultrasonic sensor
sensor_data = " "

#CAR_IP = '192.168.1.5'    # Server(Raspberry Pi) IP address
#CAR_IP = '192.168.1.6'    # Server(Raspberry Pi) IP address
#CAR_IP = '10.246.50.143'    # Server(Raspberry Pi) IP address
CAR_IP = '10.246.50.153'    # Server(Raspberry Pi) IP address

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


####################################### Neural Network definition ##############################


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ANN_MLP()

    def create(self):
        layer_size = np.int32([38400, 32, 4])
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
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',30)))
        #initial steer command set to stop
        NNsteerCommand = 3
        oldSteerCommand = NNsteerCommand+1
        try:
            print 'Start Main Thread for Deep Drive'
            time1=0
            
            #start receiver thread client to receive continuously data
            if videoClientEnable == True :
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE_IMAGE,''))

            if sensorClientEnable == True :
                 self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))
       
            while True:
                
                ############################# Manage IMAGE for Deep neural network to extract Steer Command ###############
                try:
                    # try to see if image ready
                    reply = self.sctVideoStream.reply_q.get(True,1)
                    if reply.type == ClientReply.SUCCESS:

                        #little check on frame received
                        dt=datetime.now()      
                        #print "time1 = %.1f"%((dt.microsecond-time1) / 1000)
                        time1 = dt.microsecond
                            
                        #check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
     
                        #print length as debug
                        #print 'length =' + str(len(self.sctVideoStream.lastImage))
                        
                        #decode jpg into array
                        image = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)

                        # lower half of the image
                        half_gray = image[0:120,:]

                        #cv2.imshow('image', image)
                        cv2.imshow('mlp_image', half_gray)

                        # reshape image
                        image_array = half_gray.reshape(1, 38400).astype(np.float32)

                        # neural network makes prediction
                        NNsteerCommand = self.model.predict(image_array)
                   
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
                
                # for the moment we just have DNN prediction :
                if oldSteerCommand != NNsteerCommand :
                    self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_NN',NNsteerCommand)))
                    oldSteerCommand = NNsteerCommand
                
                
        finally:
            #stop and close all client and close them
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
              
if __name__ == '__main__':
    #create Deep drive thread and strt
    DDriveThread = DeepDriveThread()
    DDriveThread.name = 'DDriveThread'
    
    #start
    DDriveThread.start()

    DDriveThread.join()

    print 'end'


