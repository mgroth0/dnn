from time import sleep
# noinspection PyUnresolvedReferences
import initFun
from loggy import log
from mutil import File

class WC_Sync_API:
    def __setitem__(self, key, value):
        File('WC/put').resolve(key).write(value)

    def __getitem__(self, item):
        File('WC/get-request').resolve(item).write('')
        response = File('WC/get-response').resolve(item)
        while not response.exists():
            sleep(0.01)
        data = response.read()
        response.delete()
        return data

WC = WC_Sync_API()

if __name__ == '__main__':
    log('testing put 1')
    WC['test1'] = 'result1'
    log('tested put 1')
    log('testing put 2')
    WC['test2'] = 'result2'
    log('tested put 2')

    log('testing get 1')
    r1 = WC['test1']
    log('tested get 1:$', r1)
    log('testing get 2')
    r2 = WC['test2']
    log('tested get 2:$', r2)

    WC['lastsave'] = 'mac'
