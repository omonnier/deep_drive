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
import pygame
from pygame.locals import *



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
sensorClientEnable = False

MIN_ANGLE    = -50
MAX_ANGLE    = 50

# STEP_CAPTURE is the minimal step we send to the car 
STEP_CAPTURE = 1

turning_offset = 0
#sampling of the main loop in s
LOOP_TIME = 0.04

#sampling of the image into vstack array (FPS record)
FPS_RECORD_TIME = 0.1

class CollectTrainingData(threading.Thread):
    
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
        
        saved_frame = 0
        total_frame = 0
        turn_angle = 0
        totalAngle = 0
        lastTotalAngle = 0
            
        #Send speed to car 
        print 'set Speed'
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',25)))
        #initial steer command set to stop
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))

        
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400), dtype=np.uint8)
        label_array = np.zeros(1, dtype=np.uint8)

        pygame.init()
        # use the window ygame to enter direction
        screen = pygame.display.set_mode((400,300))
        #pygame.display.iconify()

        key_input = pygame.key.get_pressed()

        totalRecordTime=0

        # stream video frames one by one
        try:         
            bytes=''
            frame = 1
            next_turn = 0
            last_key_pressed = 0
            record = 0

            print 'Start Main Thread to collect image'
            
            #start receiver thread client to receive continuously data
            if videoClientEnable == True :
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE_IMAGE,''))

            if sensorClientEnable == True :
                 self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #init time
            lastFrameTime = time.time()
            lastSteerTime = lastFrameTime
            
            while True:
                try:
                    time0=time.time()
                    # check queue success for image ready
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        i = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
                        
                        # select highest half of the image vertical (120/240) and half image horizontal
                        roi = i[120:240,:]
                        
                        # save streamed images
                        #cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), i)
                        
                        cv2.imshow('roi_image', roi)
                        #cv2.imshow('image', i)
                        
                        # reshape the roi image into one row array
                        temp_array = roi.reshape(1, 38400).astype(np.uint8)

                        frame += 1
                        total_frame += 1                    
                        event_handled = 0
                        
                    else:
                        print 'Error getting image :' + str(reply.data)
                        break
                    
                except Queue.Empty:
                    #queue empty most of the time because image not ready
                    pass

                #time.sleep(0.01)
                ############################# Receive input key from human driver ######################
                turn_angle = 0
                
                event_handled = 0
                event = 0
                # receive new input from human driver
                for event in pygame.event.get():
                    event_handled = 1
                if event_handled == 1:
                    # Check Key pressed
                    if event.type == KEYDOWN:
                        if event.key == K_x or event.key == K_q or event.key == K_a or event.key == K_ESCAPE:
                            print 'exit'
                            self.send_inst = False
                            record = 0
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                            break
                    
                        elif event.key == K_UP:
                            last_key_pressed = K_UP
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))
                            record = 1
                            recordTime = time.time()
                       
                        elif event.key == K_DOWN:
                            last_key_pressed = K_DOWN
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                    
                        elif event.key == K_RIGHT:
                            last_key_pressed = K_RIGHT
                            turn_angle = STEP_CAPTURE

                        elif event.key == K_LEFT:
                            last_key_pressed = K_LEFT
                            turn_angle = -STEP_CAPTURE

                        elif event.key == K_SPACE:
                            last_key_pressed = K_SPACE
                            record = 1
                            recordTime = time.time()
                            

                            
                    # In case there is an KEYUP event for Left/right
                    # Check if the other key is still down                  
                    elif event.type == KEYUP:
                        key_input = pygame.key.get_pressed()
                        if event.key == K_RIGHT:
                            if key_input[pygame.K_LEFT]:
                                last_key_pressed = K_LEFT
                                turn_angle = -STEP_CAPTURE
                            else:
                                last_key_pressed = 0
                                turn_angle = 0
                                
                        elif event.key == K_LEFT:
                            if key_input[pygame.K_RIGHT]:
                                last_key_pressed = K_RIGHT
                                turn_angle = STEP_CAPTURE
                            else:
                                last_key_pressed = 0
                                turn_angle = 0
                                
                        elif event.key == K_SPACE:
                            record = 0
                            totalRecordTime += time.time() - recordTime
                            last_key_pressed = 0

                        elif event.key == K_DOWN:
                            record = 0
                            last_key_pressed = 0
                            totalRecordTime += time.time() - recordTime
                            
                        else:
                            last_key_pressed = 0
                            turn_angle = 0
                    else:
                        last_key_pressed = 0
                        turn_angle = 0
                      
                # Turning specific handling.        
                #  Key still down ==> Continue turning right/left
                #  Key still up   ==> Continue goig back to home
                if event_handled == 0:
                    if (last_key_pressed == K_RIGHT):
                        turn_angle = STEP_CAPTURE

                    elif last_key_pressed == K_LEFT:
                        turn_angle = -STEP_CAPTURE

                #get time for steer command
                timeNow = time.time()
                timeTarget = lastFrameTime + FPS_RECORD_TIME
                if (record == 1):
                    #check if this is time to record a frame
                    if  timeNow > timeTarget:
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, np.array([totalAngle])))
                        lastFrameTime = timeNow

                #get time for steer command (Warning , the stack takes time so we redo get time here)
                timeNow = time.time()
                timeTarget = lastSteerTime + LOOP_TIME
                if timeNow > timeTarget:
                    #it s time to updat steer command
                    totalAngle += turn_angle
                    if totalAngle >= MAX_ANGLE:
                        totalAngle = MAX_ANGLE
                    elif totalAngle <= MIN_ANGLE:
                        totalAngle = MIN_ANGLE                        
                    self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',totalAngle)))
                    lastSteerTime = timeNow
                    if lastTotalAngle != totalAngle:
                        print 'turn_angle = ',totalAngle
                        lastTotalAngle = totalAngle
                
            
            # Convert image in float
            image_array = np.asarray (image_array, np.float32)

            # save training images and labels
            train = image_array[1:, :]
            train_labels = label_array[1:, :]

            # save training data as a numpy file
            np.savez('training_data_temp/test08.npz', train=train, train_labels=train_labels)


            e2 = cv2.getTickCount()
            # calculate streaming duration
            time0 = (e2 - e1) / cv2.getTickFrequency()
            print 'Streaming duration:', time0

            print(train.shape)
            print(train_labels.shape)
            print 'Total frame:', total_frame
            print 'Saved frame:', saved_frame , ' in ', totalRecordTime, ' seconds'
            print 'Dropped frame', total_frame - saved_frame

        finally:
            #stop and close all client and close them
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))
            
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
    DDCollectData = CollectTrainingData()
    DDCollectData.name = 'DDriveThread'
    
    #start
    DDCollectData.start()

    DDCollectData.join()

    print 'end'
