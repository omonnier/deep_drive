import threading
import urllib
import Queue
import struct
import socket
from commonDeepDriveDefine import *



class VideoThread(threading.Thread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(VideoThread, self).__init__()
        #create a queue for each object 
        self.cmd_q = Queue.Queue()
        self.reply_q = Queue.Queue()
        self.alive = threading.Event()
        self.alive.set()
        self.stream = None
        self.rcvBytes = ''
        self.lastImage= ''
        
        self.handlers = {
            ClientCommand.CONNECT: self._handle_CONNECT,
            ClientCommand.CLOSE: self._handle_CLOSE,
            ClientCommand.CLOSE: self._handle_STOP,
            ClientCommand.RECEIVE_IMAGE: self._handle_RECEIVE_IMAGE,
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
                self.stream = urllib.urlopen(cmd.data)
                self.reply_q.put(self._success_reply())
                return
            except IOError as e:
                pass
                #print 'Video Connect Error : ' + str(e)
                #then retry again
                

    # used to receive jpg image in client thread 
    def _handle_RECEIVE_IMAGE(self, cmd):
        while True:
            #check first if new command to stop comes in
            try:
                newCmd = self.cmd_q.get(False)
                if newCmd.type == ClientCommand.STOP:
                    return
            except Queue.Empty:
                #we should always be there
                pass
            
            try:
                #loop until image found or problem
                self.rcvBytes += self.stream.read(1024)
                #print 'rcv = ' + str(len(self.rcvBytes))
                # search for jpg image 
                a = self.rcvBytes.find('\xff\xd8')
                b = self.rcvBytes.find('\xff\xd9')
                if a!=-1 and b!=-1:
                    #image found , send it in receive queue
                    self.lastImage = self.rcvBytes[a:b+2]
                    self.reply_q.put(self._success_reply())
                    #now shift rcvbyte to manage next image
                    self.rcvBytes=self.rcvBytes[b+2:]

            except IOError as e:
                self.reply_q.put(self._error_reply(str(e)))

    def _handle_CLOSE(self, cmd):
        self.socket.close()
        
    def _handle_STOP(self, cmd):
        pass
        
    def _handle_SEND(self, cmd):
        header = struct.pack('<L', len(cmd.data))
        try:
            self.socket.sendall(header + cmd.data)
            self.reply_q.put(self._success_reply())
        except IOError as e:
            self.reply_q.put(self._error_reply(str(e)))
    
    def _handle_RECEIVE(self, cmd):
        try:
            header_data = self._recv_n_bytes(4)
            if len(header_data) == 4:
                msg_len = struct.unpack('<L', header_data)[0]
                data = self._recv_n_bytes(msg_len)
                if len(data) == msg_len:
                    self.reply_q.put(self._success_reply(data))
                    return
            self.reply_q.put(self._error_reply('Socket closed prematurely'))
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

