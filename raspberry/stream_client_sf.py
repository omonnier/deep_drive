"""
Reference:
PiCamera documentation
https://picamera.readthedocs.org/en/release-1.10/recipes2.html

"""

import io
import socket
import struct
import time
import cv2
import numpy as np


# create socket and bind host
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.1.100', 8000))
#client_socket.connect(('127.0.0.1', 8000))
print("sockect alive")
connection = client_socket.makefile('wb')

try:
    start = time.time()

    cap = cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
        
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # Our operations on the frame come here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame',frame)
            #(1,80) = (IMWRITE_JPEG_QUALITY,80 for quailty %)
            r, buff = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY, 50])
            if (r == True) :
                    for value in buff:
                        connection.write(struct.pack('<B', value))
                        
            else:
                print'conversion into jpg failed'
            connection.flush()   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                connection.write(struct.pack('<L', 0))
                break
        connection.write(struct.pack('<L', 0))
finally:
    cap.release()
    connection.close()
    client_socket.close()
    cv2.destroyAllWindows()
