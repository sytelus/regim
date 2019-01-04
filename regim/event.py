from collections import OrderedDict

class Event:
    def __init__(self):
        self._listners = OrderedDict()

    def subscribe(self, f):
        self._listners[f] = None

    def unsubscribe(self, f):
        self._listners.pop(f, None)

    def notify(self, *args,**kwargs):
        for f in self._listners.keys():
            f(*args,**kwargs) 
