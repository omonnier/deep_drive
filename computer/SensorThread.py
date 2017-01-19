import threading
import urllib
import Queue
import struct
import socket
from commonDeepDriveDefine import *



class SensorThread(threading.Thread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(SensorThread, self).__init__()
        #create a queue for each object 
        self.cmd_q = Queue.Queue()
        self.reply_q = Queue.Queue()
        self.alive = threading.Event()
        self.alive.set()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

        
        self.handlers = {
            ClientCommand.CONNECT: self._handle_CONNECT,
            ClientCommand.CLOSE: self._handle_CLOSE,
            ClientCommand.RECEIVE: self._handle_RECEIVE,
            ClientCommand.STOP: self._handle_STOP,
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
                pass
                
    def join(self, timeout=None):
        #print 'WARNING : queue stopped : ' + threading.currentThread().getName()
        self.alive.clear()
        threading.Thread.join(self, timeout)

    def _handle_CLOSE(self, cmd):
        self.socket.close()
        
    def _handle_STOP(self, cmd):
        pass
           
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
                self.connected = True
                self.reply_q.put(self._success_reply())
                return
            
            except IOError as e:
                pass
                #print 'Steer Connect Error : ' + str(e)
                #retry again


    def _handle_RECEIVE(self, cmd):
        if self.connected == True:
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
                    header_data = self._recv_n_bytes(4)
                    if len(header_data) == 4:
                        msg_len = struct.unpack('<L', header_data)[0]
                        data = self._recv_n_bytes(msg_len)
                        if len(data) == msg_len:
                            #check if we have more than 5 cm compare to previous
                            if abs(lastDataSent - data) > 5:
                            #send in reply queue the data in cm and int format
                                self.reply_q.put(self._success_reply(int(data)))
                                lastDataSent = data
                    else:
                        #for whatever reason the len does not match
                        self.reply_q.put(self._error_reply('Sensor socket misalignement'))
                        
                except IOError as e:
                    self.reply_q.put(self._error_reply(str(e)))


    def _recv_n_bytes(self, n):
        """ Convenience method for receiving exactly n bytes from self.socket
            (assuming it's open and connected).
        """
        data = ''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if chunk == '':
                break
            data += chunk
        return data        
        
    def _error_reply(self, errstr):
        return ClientReply(ClientReply.ERROR, errstr)

    def _success_reply(self, data=None):
        return ClientReply(ClientReply.SUCCESS, data)

