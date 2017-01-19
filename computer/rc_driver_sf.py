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


####################################### Neural Network definition ##############################


class NeuralNetwork(object):

    def __init__(self):
        self.model = cv2.ANN_MLP()

    def create(self):
        layer_size = np.int32([38400, 32, NN_OUTPUT_NUMBER])
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


        #create Keyboard Thread
        self.keyboardThread = keyboardThread()
        self.keyboardThread.name = 'keyboardThread'
        self.keyboardThread.start()
        
        #connect All client
        self.ConnectClient()
             
        # create neural network
        self.model = NeuralNetwork()
        self.model.create()

        #init runnin gprediction average values with null angle. MANDATORY for the first sample
        self.predictionValuesToAverage  = np.zeros(NB_SAMPLE_RUNNING_AVERAGE_PREDICTION, dtype=np.int)
        self.predictionIndex = 0

        #reset the initial angle command
        self.steerAngleCommand = 0

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
        record=0
        totalRecordTime = 0
        recordTime = 0
        lastKeypressed = 0
        turn_angle = 0
        steerKeyboardAngle = 0
        lastSteerKeyboardAngle =0
        saved_frame = 0
        total_frame = 0

        image_record_array = np.zeros((1, 38400), dtype=np.uint8)
        label_record_array = np.zeros(1, dtype=np.uint8)

        
        #init timing  
        lastSteerControlTime = time.time()
        lastSteerKeyboardTime = time.time()
        lastFrameTime = time.time()
            
        #initial steer command set to stop
        try:
            print 'Start Main Thread for Deep Drive'
            
            #start receiver thread client to receive continuously data
            if videoClientEnable == True :
                self.sctVideoStream.cmd_q.put(ClientCommand(ClientCommand.RECEIVE_IMAGE,''))

            if sensorClientEnable == True :
                 self.sctSensor.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #start keyboard thread to get keyboard inut
            self.keyboardThread.cmd_q.put(ClientCommand(ClientCommand.RECEIVE,''))

            #start car to be able to see additioanl data
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('SPEED',25)))
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',0)))
            
            
            while True:
                ############################# Manage IMAGE for Deep neural network to extract Steer Command ###############
                try:
                    # try to see if image ready
                    reply = self.sctVideoStream.reply_q.get(False)
                    if reply.type == ClientReply.SUCCESS:

                        #print length as debug
                        #print 'length =' + str(len(self.sctVideoStream.lastImage))
                        
                        #decode jpg into array
                        image = cv2.imdecode(np.fromstring(self.sctVideoStream.lastImage, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)

                        # lower half of the image
                        half_gray = image[120:240,:]

                        #cv2.imshow('image', image)
                        cv2.imshow('mlp_image', half_gray)

                        #check if we want to stop autonomous driving
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        # reshape image
                        image_array_byte = half_gray.reshape(1, 38400).astype(np.uint8)
                        
                        # Convert image in float 
                        image_array = np.asarray (image_array_byte, np.float32)
                        #print image_array;

                        # neural network makes prediction
                        NNsteerCommand = self.model.predict(image_array)

                        #WARNING QUICK AND DIRTY FIX BEFORE WE UNDERSTAND WHY WE HAVE TOO MUCH 0 prediction
                        if NNsteerCommand != 0:
                            # fill average angle table based on prediction
                            self.predictionValuesToAverage[self.predictionIndex] = self.sctSteer.NNprediction2Angle(NNsteerCommand)
                            self.predictionIndex += 1
                            if self.predictionIndex >= NB_SAMPLE_RUNNING_AVERAGE_PREDICTION:
                                self.predictionIndex = 0

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
                            record = 1
                            turn_angle = STEP_CAPTURE
                            
                        elif keyPressed == 'left':
                            record = 1
                            turn_angle = -STEP_CAPTURE

                        elif keyPressed == 'up':
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','forward')))
                            turn_angle = 0

                        elif keyPressed == 'down':
                            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
                            turn_angle = 0
                            
                        elif keyPressed == 'space':
                            record = 1
                            
                        elif keyPressed == 'none':
                            turn_angle = 0
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
                        image_record_array = np.vstack((image_record_array, image_array_byte))
                        label_record_array = np.vstack((label_record_array, np.array([self.steerAngleCommand])))
                        lastFrameTime = timeNow
                else:
                    #record the time if recorTime exist
                    if recordTime != 0:
                        totalRecordTime += (time.time() - recordTime)
                        recordTime = 0

                #get time and manage ster from keyboard  
                timeNow = time.time()
                if timeNow > (lastSteerKeyboardTime + STEERING_KEYBOARD_SAMPLING_TIME):
                    #it s time to update steer command
                    steerKeyboardAngle += turn_angle
                    if steerKeyboardAngle >= MAX_ANGLE:
                        steerKeyboardAngle = MAX_ANGLE
                    elif steerKeyboardAngle <= MIN_ANGLE:
                        steerKeyboardAngle = MIN_ANGLE
                    lastSteerKeyboardTime = timeNow
                    


                ############### Control the Car with all the input we can have ####################

                #handle now all input to determine the control of the car
                if record == 0:
                    #no record on going ---> we believe prediction !
                    #send control command according to sampling dedicated for it
                    timeNow = time.time()
                    if timeNow > (lastSteerControlTime + STEERING_PREDICTION_SAMPLING_TIME):
                        #no record on going ---> we believe prediction !
                        prediction = np.sum (self.predictionValuesToAverage, dtype=int) / NB_SAMPLE_RUNNING_AVERAGE_PREDICTION
                        #print 'prediction = ' + str(self.predictionValuesToAverage) + ' , average_value = ' + str(steerKeyboardAngle)
                        if self.steerAngleCommand != prediction:
                            print 'prediction Angle= '  + str(prediction)
                        #send command
                        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',prediction)))
                        self.steerAngleCommand = prediction
                        #reset the steerKeyboard angle to the latest angle to start from it if correctio needed
                        steerKeyboardAngle = self.steerAngleCommand
                        lastSteerControlTime = timeNow
                else:

                    #send control command according to sampling dedicated for it
                    timeNow = time.time()
                    if timeNow > (lastSteerControlTime + STEERING_KEYBOARD_SAMPLING_TIME):
                        self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_ANGLE',steerKeyboardAngle)))
                        #we DO NOT believe prediction and use keyboard correction 
                        self.steerAngleCommand = steerKeyboardAngle
    
                        #check and print if prediction is good enough compared to Forced angle                                                    
                        prediction = np.sum (self.predictionValuesToAverage, dtype=int) / NB_SAMPLE_RUNNING_AVERAGE_PREDICTION
                        if abs(prediction - steerKeyboardAngle) >  MAX_KEYBOARD_DELTA_ANGLE_TO_PREDICTION:
                            #for now we just print out this potential problem
                            print 'WARNING, prediction still far from forced control'

                        #print out only if changed
                        if lastSteerKeyboardAngle != steerKeyboardAngle:
                            print 'FORCED turn_angle = ',steerKeyboardAngle
                            lastSteerKeyboardAngle = steerKeyboardAngle
                            
                        lastSteerControlTime = timeNow

                
        finally:
            print 'ending Deep Driver'
            if totalRecordTime !=0:
                
                # Convert image in float
                image_record_array = np.asarray (image_array, np.float32)

                # save training images and labels
                train = image_record_array[1:, :]
                train_labels = label_record_array[1:, :]

                # save training data as a numpy file
                np.savez('training_data_temp/test08.npz', train=train, train_labels=train_labels)

                print(train.shape)
                print(train_labels.shape)
                print 'Total frame:', total_frame
                print 'Saved frame:', saved_frame , ' in ', totalRecordTime, ' seconds'
            
            #stop and close all client and close them
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','stop')))
            self.sctSteer.cmd_q.put(ClientCommand(ClientCommand.SEND, ('STEER_COMMAND','home')))
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
            #and make sure all of them ended properly
            self.sctVideoStream.join()
            self.sctSteer.join()
            self.sctSensor.join()
            self.keyboardThread.join()
            print 'Deep Driver Done'
            
if __name__ == '__main__':
    #create Deep drive thread and strt
    DDriveThread = DeepDriveThread()
    DDriveThread.name = 'DDriveThread'
    
    #start
    DDriveThread.start()

    DDriveThread.join()
    print 'end'


