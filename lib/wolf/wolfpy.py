import logging
import pdb

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

from lib.boot.loggy import log
from lib.misc.mutil import File, strcmp, err
from mlib.boot.bootutil import ismac

class WolfPy:
    MADE = False
    def __init__(self):
        # err('dont use this for now')
        if self.MADE:
            raise Exception('just use one wl kernel')
        self.MADE = True
        if ismac():
            self.session = WolframLanguageSession(
                kernel_loglevel=logging.DEBUG,
                # kernel_loglevel=logging.FATAL,
            )
        else:
            self.session = WolframLanguageSession(
                kernel_loglevel=logging.DEBUG,
                # kernel_loglevel=logging.FATAL,
                                                  kernel='/home/matt/WOLFRAM/Executables/WolframKernel')

    def eval(self, s):
        ev = self.session.evaluate_wrap_future(wl.UsingFrontEnd(s))
        ev = ev.result()
        if ev.messages is not None:
            for m in ev.messages:
                err(m)
        return ev.result

    def __getattr__(self, name):
        return weval(name)

    def export(self, filename, wlstuff, *args):
        log(f'exporting to {filename}..')
        self.eval(wl.Export(filename, wlstuff, *args))
        log(f"created {filename}")

    def cloud_get(self, name):
        return self.eval(wl.CloudGet(name))

    def cloud_put(self, name, value):
        return self.eval(wl.CloudPut(name, value))

    def cloud_export(self, wlstuff, *args, format="PNG", url=None):
        myargs = []
        if url is not None:
            myargs.append(wl.CloudObject(url))
        exp = wl.CloudExport(wlstuff, format, *myargs, *args)
        co = self.eval(exp)
        return co

    def cloud_deploy(self, wlstuff, url=None, public=False, *args):
        myargs = []
        if url is not None:
            myargs.append(url)
        if public:
            myargs.append(wl.Rule(wl.Permissions, "Public"))
        exp = wl.CloudDeploy(wlstuff, *myargs, *args)
        co = self.eval(exp)
        return co

    def push_file(self, fromm):
        return self.copy_file(fromm, fromm)

    def copy_file(self, fromm, to, permissions='Private'):
        tos = File(to).names()
        # if os.path.isabs(to):
        flag = True
        while flag:
            if strcmp(tos[0], 'mitili', ignore_case=True):
                flag = False
            tos = tos[1:]

        return self.eval(wl.CopyFile(
            File(fromm).abspath,
            wl.CloudObject(wl.FileNameJoin(tos), wl.Rule(wl.Permissions, permissions))
        ))


WOLFRAM = WolfPy()
def weval(arg): return WOLFRAM.eval(arg)
