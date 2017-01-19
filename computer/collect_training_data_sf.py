import cv2
import numpy as np
import math
import threading
import Queue
import time
from commonDeepDriveDefine import *
from KeyboardThread import *
from SteerThread import *
from VideoThread import *
from SensorThread import *


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

        #create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'keyboardThread'
        self.keyboardThread.start()
        
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
        lastTotalAngle = 0
        totalRecordTime=0
        frame = 1
        record = 0
        bytes=''
        turn_angle = 0
        totalAngle = 0
        lastkeypressed = 0
        recordTime = 0
        totalRecordTime = 0
                     
            
        #Send speed to car 
        print 'Enter main thread to collect data'
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',28)))
        #initial steer command set to stop
        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))
        
        image_array = np.zeros((1, 38400), dtype=np.uint8)
        label_array = np.zeros(1, dtype=np.uint8)

        # stream video frames one by one
        try:         

            print 'Start Main Thread to collect image'
            
            #start receiver thread client to receive continuously data
            if videoClientEnable == True :
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE_IMAGE,''))

            if sensorClientEnable == True :
                 self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #start keyboard thread to get keyboard inut
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #init time
            lastFrameTime = time.time()
            lastSteerTime = lastFrameTime



            while True:
                try:
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

                        #check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
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

                ######################## Get control from the keyboard if any #########################
                try:
                    # keyboard queue filled ?
                    reply = self.keyboardThread.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:
                        #new keyboard input found
                        keyPressed = reply.data
                        print 'key Pressed = ' , keyPressed
                        
                        if keyPressed == 'exit':
                            record = 0
                            turn_angle = 0
                            if recordTime != 0:
                                totalRecordTime += (time.time() - recordTime)
                            #get out of the loop
                            break
                        
                        elif keyPressed == 'right':
                            turn_angle = STEP_CAPTURE
                            
                        elif keyPressed == 'left':
                            turn_angle = -STEP_CAPTURE

                        elif keyPressed == 'up':
                            record = 1
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))

                        elif keyPressed == 'down':
                            record = 0
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                            turn_angle = 0
                            
                        elif keyPressed == 'space':
                            record = 1
                            
                        elif keyPressed == 'none':
                            turn_angle = 0
                            if lastkeypressed == 'space':
                                #this was a free space record then we canstop record
                                record = 0                                                         
                        else :
                            #none expeted key is pressed
                            print 'Error , another key seems to exist ???'
                            
                        # record lastkey that can be use for consecutive command action
                        lastkeypressed = keyPressed
                            
                    else:
                        print 'Error getting keyboard input :' + str(reply.data)
                        break             
                except Queue.Empty:
                    #queue empty most of the time because keyboard not hit
                    pass
                
                #See now if we have to record or not the frame into vstack memory
                timeNow = time.time()
                if (record == 1):
                    #start recording time
                    if recordTime == 0:
                        recordTime = time.time()
                    
                    #check if this is time to record a frame
                    if  timeNow > (lastFrameTime + FPS_RECORD_TIME):
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, np.array([totalAngle])))
                        lastFrameTime = timeNow
                else:
                    #record the time if recorTime exist
                    if recordTime != 0:
                        totalRecordTime += (time.time() - recordTime)
                        recordTime = 0

                #get time for steer command and apply it if done
                timeNow = time.time()
                if timeNow > (lastSteerTime + STEERING_KEYBOARD_SAMPLING_TIME):
                    #it s time to update steer command
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
                
            if totalRecordTime !=0:            
                # Convert image in float
                image_array = np.asarray (image_array, np.float32)

                # save training images and labels
                train = image_array[1:, :]
                train_labels = label_array[1:, :]

                # save training data as a numpy file
                np.savez('training_data_temp/test08.npz', train=train, train_labels=train_labels)

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
                self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.STOP))
                self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.CLOSE))
                #let 1 second for process to close
                time.sleep(2)
                self.sctVideoStream.join()
                self.sctSteer.join()
                self.sctSensor.join()
                self.keyboardThread.join()

if __name__ == '__main__':

    #create Deep drive thread and strt
    DDCollectData = CollectTrainingData()
    DDCollectData.name = 'DDriveThread'
    
    #start
    DDCollectData.start()

    DDCollectData.join()

    print 'end'
