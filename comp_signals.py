from dataclasses import dataclass

import json
from time import time

from mlib.boot import log

@dataclass
class CompSignaler:
    id: int
    def __post_init__(self):
        self._start_time = None


    def send_start_signal(self):
        self._start_time = time()
        rrr = self._send_signal(start_time=self._start_time, end_time=None)
        log('SENT START SIGNAL')
        return rrr

    def send_end_signal(self):
        return self._send_signal(start_time=self._start_time, end_time=time())

    def _send_signal(self, start_time, end_time):
        with open(f'/Users/matt/Desktop/registered/signal/comp/{time()}.json', 'w') as f:
            f.write(json.dumps({
                'id'     : self.id,
                'started': start_time,
                'ended'  : end_time
                # 'message': 'STARTED'
            }))
