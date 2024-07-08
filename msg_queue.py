from collections import deque

class MessageQueue:
    def __init__(self, len=5):
        self.queue = deque(maxlen=len)
    
    def add_msg(self, msg):
        self.queue.append(msg)
    
    def get_msg(self):
        if self.queue:
            # step t
            return self.queue[-1]
        else:
            return None
        
    def delete_newest_msg(self):
        if self.queue:
            self.queue.pop()
        else:
            print("Queue is empty!")