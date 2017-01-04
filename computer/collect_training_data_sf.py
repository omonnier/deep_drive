
import numpy as np
import cv2
import pygame
import urllib
from pygame.locals import *
from socket import * 

#### client to control sf car
ctrl_cmd = ['forward', 'backward', 'left', 'right', 'stop', 'read cpu_temp', 'home', 'distance', 'x+', 'x-', 'y+', 'y-', 'xy_home']


#HOST = '192.168.1.5'    # Server(Raspberry Pi) IP address
#HOST = '192.168.1.6'    # Server(Raspberry Pi) IP address
#HOST = '10.246.50.143'    # Server(Raspberry Pi) IP address
HOST = '10.246.51.95'    # Server(Raspberry Pi) IP address
#HOST = '169.254.77.149'    # Server(Raspberry Pi) IP address

PORT = 21567
ADDR = (HOST, PORT)



    

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
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        self.temp_label = np.zeros((1, 4), 'float')
        
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
        if (command is 'f'):
            print("Forward")
            self.tcpCliSock.sendall('home')
            self.tcpCliSock.sendall('forward')
        elif (command is 's'):
            print("stop")
            self.tcpCliSock.sendall('home')
            self.tcpCliSock.sendall('stop')
        elif (command is 'r'):
            print("right")
            self.tcpCliSock.sendall('right')
        elif (command is 'l'):
            print("left")
            self.tcpCliSock.sendall('left')  

        self.oldSteerCommand = command
        


    
    def collect_image(self):

        saved_frame = 0
        total_frame = 0

        # collect images for training
        print 'Start collecting images...'
        e1 = cv2.getTickCount()
        image_array = np.zeros((1, 38400))
        label_array = np.zeros((1, 4), 'float')

        key_input = pygame.key.get_pressed()

        # init car and speed
        print 'set Speed'
        self.tcpCliSock.send('speed' + str(8))  # Send the speed data

        # stream video frames one by one
        try:         
            bytes=''
            frame = 1
            stream=urllib.urlopen('http://' + HOST + ':8080/?action=stream')
            while self.send_inst:
                bytes += stream.read(1024)
                a = bytes.find('\xff\xd8')
                b = bytes.find('\xff\xd9')
                
                if a!=-1 and b!=-1:
                    jpg = bytes[a:b+2]
                    
                    i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    
                    # select highest half of the image vertical (120/240) and half image horizontal
                    roi = i[0:120,:]
                    
                    # save streamed images
                    #cv2.imwrite('training_images/frame{:>05}.jpg'.format(frame), i)
                    
                    cv2.imshow('roi_image', roi)
                    #cv2.imshow('image', i)
                    
                    # reshape the roi image into one row array
                    temp_array = roi.reshape(1, 38400).astype(np.float32)
                    
                    frame += 1
                    total_frame += 1                    
                    
                    # receive new input from human driver
                    for event in pygame.event.get():
                        if event.type == KEYDOWN:
                            key_input = pygame.key.get_pressed()
                            

                    if key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'exit'
                        self.send_inst = False
                        self.sendSteerCommand('s')
                        break
                    
                    elif key_input[pygame.K_UP]:
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[2]))
                        self.sendSteerCommand('f')
                       
                
                    elif key_input[pygame.K_DOWN]:
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[3]))
                        self.sendSteerCommand('s')
                    
                    elif key_input[pygame.K_RIGHT]:
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[1]))
                        saved_frame += 1
                        self.sendSteerCommand('r')

                    elif key_input[pygame.K_LEFT]:
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[0]))
                        saved_frame += 1
                        self.sendSteerCommand('l')

                    '''
                    # KEYUP management not needed for now
                    elif event.type == pygame.KEYUP and key_input[pygame.K_UP] == 0:
                        print 'stop'
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[4]))
                        self.tcpCliSock.send('stop')
                        
                    elif ((key_input[pygame.K_LEFT] == 0) and (key_input[pygame.K_RIGHT] == 0)):
                        print 'home'
                        saved_frame += 1
                        image_array = np.vstack((image_array, temp_array))
                        label_array = np.vstack((label_array, self.k[3]))
                        self.tcpCliSock.send('home')

                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print 'exit'
                        self.send_inst = False
                        self.tcpCliSock.send('home')
                        self.tcpCliSock.send('stop')
                        break
                    '''


                    del stream 
                    stream=urllib.urlopen('http://10.246.51.95:8080/?action=stream')
                    bytes=''
                    
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
