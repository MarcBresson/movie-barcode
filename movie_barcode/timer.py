from datetime import datetime
 
class Timer(object):
    def __init__(self, total: int):
        self.total = total
        self.start()

    def start(self):
        """
        starts (or restart) the timer.
        """
        self.start = datetime.now()

        return self

    def remains(self, done: int):
        """
        returns the estimated time.
        """
        now  = datetime.now()
        
        left_estimation = (self.total - done) * (now - self.start) / done
        sec = int(left_estimation.total_seconds())
        if sec < 60:
           return f"{sec} seconds"
        else:
           return f"{int(sec / 60)} minutes"
