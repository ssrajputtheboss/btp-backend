class Logger():
    def __init__(self , active=True):
        self.active = active 
    def log(self, msg):
        if(self.active):
            print(msg)

logger = Logger(False)