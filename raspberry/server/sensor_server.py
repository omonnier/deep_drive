#!/usr/bin/python


# -----------------------
# Import required Python libraries
# -----------------------
import time
import RPi.GPIO as GPIO
import struct
from socket import *

HOST = ''           # The variable of HOST is null, so the function bind( ) can be bound to all valid addresses.
PORT = 8002
BUFSIZ = 1024       # Size of the buffer
ADDR = (HOST, PORT)

tcpSerSock = socket(AF_INET, SOCK_STREAM)    # Create a socket.
tcpSerSock.bind(ADDR)    # Bind the IP address and port number of the server. 
tcpSerSock.listen(5)     # The parameter of listen() defines the number of connections permitted at one time. Once the 
                         # connections are full, others will be rejected.
                         
# -----------------------
# Define some functions
# -----------------------

def measure():
  # This function measures a distance
  GPIO.output(GPIO_TRIGGER, True)
  time.sleep(0.00001)
  GPIO.output(GPIO_TRIGGER, False)
  start = time.time()

  while GPIO.input(GPIO_ECHO)==0:
    start = time.time()

  while GPIO.input(GPIO_ECHO)==1:
    stop = time.time()

  elapsed = stop-start
  distance = (elapsed * 34300)/2

  return distance

def measure_average():
  # This function takes 3 measurements and
  # returns the average.
  distance1=measure()
  time.sleep(0.1)
  distance2=measure()
  time.sleep(0.1)
  distance3=measure()
  distance = distance1 + distance2 + distance3
  #distance = distance / 3
  distance = min(distance1,distance2,distance3)
  return distance

# -----------------------
# Main Script
# -----------------------

# Use BCM GPIO references
# instead of physical pin numbers
GPIO.setmode(GPIO.BCM)

# Define GPIO to use on Pi
GPIO_TRIGGER = 23
GPIO_ECHO    = 24

print "Ultrasonic Measurement"

# Set pins as output and input
GPIO.setup(GPIO_TRIGGER,GPIO.OUT)  # Trigger
GPIO.setup(GPIO_ECHO,GPIO.IN)      # Echo

# Set trigger to False (Low)
GPIO.output(GPIO_TRIGGER, False)

# Wrap main content in a try block so we can
# catch the user pressing CTRL-C and run the
# GPIO cleanup function. This will also prevent
# the user seeing lots of unnecessary error
# messages.
try:
  while True:
    print 'Waiting for connection...'
    # Waiting for connection. Once receiving a connection, the function accept() returns a separate 
    # client socket for the subsequent communication. By default, the function accept() is a blocking 
    # one, which means it is suspended before the connection comes.
    tcpCliSock, addr = tcpSerSock.accept() 
    print '...connected from :', addr     # Print the IP address of the client connected with the server.
    
    while True:
      #stat measure
      distance = measure_average()
      if distance > 300 :
        #strange value ... do not return it
        continue

      distance = int(distance)

      print "Send distance = " + str(distance)
      try:
        #send packing of length of distance
        tcpCliSock.send(struct.pack('<L', len(str(distance))))
        #send the string distance"
        tcpCliSock.send(str(distance))
      except IOError as e:
        print str(e)
        break



except KeyboardInterrupt:
  # User pressed CTRL-C
  # Reset GPIO settings
  GPIO.cleanup()
