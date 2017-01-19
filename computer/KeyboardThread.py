import threading
import Queue
from commonDeepDriveDefine import *
import pygame
from pygame.locals import *

class keyboardThread(threading.Thread):
    """ Implements the threading.Thread interface (start, join, etc.) and
        can be controlled via the cmd_q Queue attribute. Replies are placed in
        the reply_q Queue attribute.
    """
    def __init__(self):
        super(keyboardThread, self).__init__()
        #create a queue for each object 
        self.cmd_q = Queue.Queue()
        self.reply_q = Queue.Queue()
        self.alive = threading.Event()
        self.alive.set()

        
        self.handlers = {
            ClientCommand.RECEIVE: self._handle_RECEIVE,
            ClientCommand.STOP: self._handle_STOP,
        }
        


    def run(self):
        pygame.init()
        # use the window ygame to enter direction
        screen = pygame.display.set_mode((400,300))
        #pygame.display.iconify()
        
        while self.alive.isSet():
            try:
                # block for all command and wait on it
                cmd = self.cmd_q.get(True,1)
                #print threading.currentThread().getName() + 'CMD = ' + str(cmd.type)
                self.handlers[cmd.type](cmd)                
            except (Queue.Empty,KeyError) :
                #no process of CMD queue.empty because it is a regular case
                continue
                
    def join(self, timeout=None):
        #print 'WARNING : queue stopped : ' + threading.currentThread().getName()
        self.alive.clear()
        threading.Thread.join(self, timeout)


    def _handle_STOP(self, cmd):
        pass

    def _handle_CLOSE(self, cmd):
        pass
        
    def _handle_RECEIVE(self, cmd):
        while True:
            #check first if new command to stop comes in
            try:
                newCmd = self.cmd_q.get(False)
                if newCmd.type == ClientCommand.STOP:
                    return
            except Queue.Empty as message:
                #we should always be there
                pass
            
            try:
                # receive new input from human driver
                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        if event.key == K_RIGHT:
                            self.reply_q.put(self._success_reply('right'))

                        elif event.key == K_LEFT:
                            self.reply_q.put(self._success_reply('left'))

                        elif event.key == K_SPACE:
                            self.reply_q.put(self._success_reply('space'))

                        elif event.key == K_DOWN:
                            self.reply_q.put(self._success_reply('down'))

                        elif event.key == K_UP:
                            self.reply_q.put(self._success_reply('up'))

                        elif event.key == K_ESCAPE or event.key == K_q or event.key == K_a :
                            self.reply_q.put(self._success_reply('exit'))

                    elif event.type == KEYUP:
                        key_input = pygame.key.get_pressed()
                        #when keyup, only test if all keys are UP
                        if ((key_input[pygame.K_RIGHT] == 0) and
                        (key_input[pygame.K_LEFT] == 0) and
                        (key_input[pygame.K_UP] == 0) and
                        (key_input[pygame.K_DOWN] == 0) and
                        (key_input[pygame.K_SPACE] == 0)):
                            self.reply_q.put(self._success_reply('none'))
                            
            except IOError as e:
                pass
                #print 'Steer Connect Error : ' + str(e)
                #retry again
        
    def _error_reply(self, errstr):
        return ClientReply(ClientReply.ERROR, errstr)

    def _success_reply(self, data=None):
        return ClientReply(ClientReply.SUCCESS, data)

