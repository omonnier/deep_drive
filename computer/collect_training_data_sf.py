
import numpy as np
import cv2
import pygame
import urllib
from pygame.locals import *
from socket import *
from datetime import datetime

#### client to control sf car
ctrl_cmd = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']


#HOST = '192.168.1.5'    # Server(Raspberry Pi) IP address
#HOST = '192.168.1.6'    # Server(Raspberry Pi) IP address
#HOST = '10.246.50.143'    # Server(Raspberry Pi) IP address
HOST = '10.246.50.29'    # Server(Raspberry Pi) IP address
#HOST = '169.254.77.149'    # Server(Raspberry Pi) IP address

PORT = 8001
ADDR = (HOST, PORT)

MIN_ANGLE    = -50
MAX_ANGLE    = 50

# STEP_CAPTURE should always be lower than STEP_REPLAY 
STEP_CAPTURE = 2
STEP_REPLAY  = 5

turning_offset = 0

class CollectTrainingData(object):
    
    def __init__(self):
        # accept a single connection
        self.send_inst = True
        self.oldSteerCommand = 's'

        print('end PC server connection ... Start now Client connection to car server')
        # car connection
        self.tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
        self.tcpCliSock.connect(ADDR)                    # Connect with the server
        print('connected to car')

        # create labels
        self.k = np.zeros((15, 15), np.uint8)
        for i in range(15):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 15), np.uint8)
        
        pygame.init()
        # use the window ygame to enter direction
        screen = pygame.display.set_mode((400,300))
        #pygame.display.iconify()
        self.collect_image()


    def sendSteerCommand(self,command):
        '''
        if (command is self.oldSteerCommand):
            return
        else:
        '''
        global turning_offset

        if (command is 'f'):
            print("Forward")
            #self.tcpCliSock.sendall('home')
            self.tcpCliSock.sendall('forward>')
        elif (command is 's'):
            print("stop")
            self.tcpCliSock.sendall('home>')
            self.tcpCliSock.sendall('stop>')
		# Handle right/left commands
        elif (command is 'r'):
            print("right")
            self.tcpCliSock.sendall('right>')
        elif (command is 'l'):
            print("left")
            self.tcpCliSock.sendall('left>')
		
        # Handle turn using angle
        #  => Positive offset means right offset
        #  => Negative offset means left offset
        elif command[0:5] == 'turn=':
            print(command )
            self.tcpCliSock.sendall(command)
				
        self.oldSteerCommand = command
        


    
    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print 'Start collecting images...'
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400), dtype=np.uint8)
        label_array = np.zeros(1, dtype=np.uint8)

        key_input = pygame.key.get_pressed()

        # init car and speed
        print 'set Speed'
        self.tcpCliSock.send('speed' + str(20))  # Send the speed data

        # stream video frames one by one
        try:         
            bytes=''
            frame = 1
            next_turn = 0
            last_key_pressed = 0
            turn_angle = 0
            record = 0
			
            stream=urllib.urlopen('http://' + HOST + ':8000/?action=stream')
            dt=datetime.now()
            time1=dt.microsecond
            while self.send_inst:
                bytes += stream.read(1024)
                a = bytes.find('\xff\xd8')
                b = bytes.find('\xff\xd9')
                
                if a!=-1 and b!=-1:

                    #little check on frame received
                    dt=datetime.now()                    
                    #print "time1 = %.1f"%((dt.microsecond-time1) / 1000)
                    time1=((dt.microsecond-time1) / 1000)
                    if (time1 > 80):
                        print "frame received later than expected = %.1f"%(time1)
                    time1=dt.microsecond
                    
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    
                    i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    
                    # select highest half of the image vertical (120/240) and half image horizontal
                    roi = i[0:120,:]
                    
                    # save streamed images
                    #cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), i)
                    
                    cv2.imshow('roi_image', roi)
                    #cv2.imshow('image', i)
                    
                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.uint8)

                    frame += 1
                    total_frame += 1                    
                    event_handled = 0

                    # receive new input from human driver
                    for event in pygame.event.get():
                        event_handled = 1

                    # Check Key pressed
                    if event.type == KEYDOWN:
                        if event.key == K_x or event.key == K_q or event.key == K_a or event.key == K_ESCAPE:
                            print 'exit'
                            self.send_inst = False
                            self.sendSteerCommand('s')
                            break
                    
                        elif event.key == K_UP:
                            last_key_pressed = K_UP
                            saved_frame += 1
                            self.sendSteerCommand('f')
                            record = 1
                       
                        elif event.key == K_DOWN:
                            last_key_pressed = K_DOWN
                            saved_frame += 1
                            image_array = np.vstack((image_array, temp_array))
                            label_array = np.vstack((label_array, np.array([255])  ))
                            self.sendSteerCommand('s')
                    
                        elif event.key == K_RIGHT:
                            last_key_pressed = K_RIGHT
                            if (turn_angle < MAX_ANGLE ): 
                                turn_angle += STEP_CAPTURE
                            self.sendSteerCommand('turn=%d>' % (turn_angle) )

                        elif event.key == K_LEFT:
                            last_key_pressed = K_LEFT
                            if (turn_angle > MIN_ANGLE ):
                                turn_angle -= STEP_CAPTURE
                            self.sendSteerCommand('turn=%d>' % (turn_angle) )
                            
                    # In case there is an KEYUP event for Left/right
                    # Check if the other key is still down                  
                    elif event.type == KEYUP:
                        key_input = pygame.key.get_pressed()
                        if event.key == K_RIGHT:
                            if key_input[pygame.K_LEFT]:
                                last_key_pressed = K_LEFT
                            else:
                                last_key_pressed = 0
                                
                        elif event.key == K_LEFT:
                            if key_input[pygame.K_RIGHT]:
                                last_key_pressed = K_RIGHT
                            else:
                                last_key_pressed = 0
                        else:
                            last_key_pressed = 0
                    else:
                        last_key_pressed = 0
                          
                    # Turning specific handling.        
                    #  Key still down ==> Continue turning right/left
                    #  Key still up   ==> Continue goig back to home
                    if event_handled == 0:
                        if (last_key_pressed == K_RIGHT):
                            if (turn_angle < MAX_ANGLE ): 
                                turn_angle += STEP_CAPTURE
                            self.sendSteerCommand('turn=%d>' % (turn_angle) )

                        elif last_key_pressed == K_LEFT:
                            if (turn_angle > MIN_ANGLE ):
                                turn_angle -= STEP_CAPTURE
                            self.sendSteerCommand('turn=%d>' % (turn_angle) )

                    if (record == 1):
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, np.array([turn_angle ])))

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
            print 'Saved frame:', saved_frame
            print 'Dropped frame', total_frame - saved_frame

        finally:
            # stop car server/client
            self.sendSteerCommand('s')
            self.tcpCliSock.close()

if __name__ == '__main__':

    CollectTrainingData()
