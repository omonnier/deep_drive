import threading
import urllib
import Queue
import struct
import socket
from commonDeepDriveDefine import *

class SteerThread(threading.Thread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(SteerThread, self).__init__()
        #create a queue for each object 
        self.cmd_q = Queue.Queue()
        self.reply_q = Queue.Queue()
        self.alive = threading.Event()
        self.alive.set()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        self.handlers = {
            ClientCommand.CONNECT: self._handle_CONNECT,
            ClientCommand.CLOSE: self._handle_CLOSE,
            ClientCommand.SEND: self._handle_SEND,
            ClientCommand.STOP: self._handle_STOP
        }


    def run(self):
        while self.alive.isSet():
            try:
                # block for all command and wait on it
                cmd = self.cmd_q.get(True,1)
                #print threading.currentThread().getName() + 'CMD = ' + str(cmd.type)
                self.handlers[cmd.type](cmd)
            except Queue.Empty:
                #no process of CMD queue.empty because it is a regular case
                continue
                
    def join(self, timeout=None):
        #print 'WARNING : queue stopped : ' + threading.currentThread().getName()
        self.alive.clear()
        threading.Thread.join(self, timeout)

    def NNprediction2Angle(self,NNprediction):
        return int((NNprediction * STEP_REPLAY) - MAX_ANGLE)
    
    def _handle_CONNECT(self, cmd):
        # try connection ntil it succeed or close sent
        while True:
            #check if new command comes in
            try:  
                newCmd = self.cmd_q.get(False)
                if newCmd.type == ClientCommand.STOP:
                    return
            except Queue.Empty:
                #we should always be there
                pass
            
            try:
                self.socket.connect((cmd.data[0], cmd.data[1]))
                self.reply_q.put(self._success_reply())
                return
            except IOError as e:
                pass
                #print 'Steer Connect Error : ' + str(e)
                #retry again


    def _handle_CLOSE(self, cmd):
        self.socket.close()
        
    def _handle_STOP(self, cmd):
        self.socket.sendall('stop' + '>')
        
    def _handle_SEND(self, cmd):
        if cmd.data[0] is 'SPEED':
            self.socket.sendall('speed' + str(cmd.data[1]) + '>')
            
        elif cmd.data[0] is 'STEER_COMMAND':
            self.socket.sendall(cmd.data[1] + '>')
            
        elif cmd.data[0] is 'STEER_ANGLE':
	    command = 'turn=' + str(cmd.data[1]) + '>'
            self.socket.sendall(command)
            
        else:
            print 'Steer Command unknown' + str()

        
    def _error_reply(self, errstr):
        return ClientReply(ClientReply.ERROR, errstr)

    def _success_reply(self, data=None):
        return ClientReply(ClientReply.SUCCESS, data)

