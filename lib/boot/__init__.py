from . import bootfun



def miniconda():
    if ismac():
        return '/Users/matt/miniconda3'
    else:
        return '/home/matt/miniconda3'
def mitiliHome():
    if ismac():
        return pwd()
    else:
        return '/home/matt/mitili'
def checkWL():
    STATUS_FILE = File(pwd()+'/WC/status')
    SIGNAL_FILE = File(pwd()+'/WC/signal')
    if not STATUS_FILE.exists() or STATUS_FILE.msecs() < time.time() - 2:
        if SIGNAL_FILE.exists():
            SIGNAL_FILE.delete()
        pid = os.fork()
        if pid==0: # new process



            # very interesting, and it worked, but overly complicated... why not just use an Wolfram APIFun as a more formal server?
            # also, don't need this right now until I have a VM again.
            # os.system(miniconda()+"/bin/python3 "+mitiliHome()+"/WC/wc.py"
            #           )





            os._exit(0)
# print('in initFun for ' + str(os.getpid()) + "|" + sys.argv[0])



# eventually i think this is supposed to run on both systems so a flag is sent to the cloud for whichver most recently updated the data. but for now I don't wanna figure out which files should and which shouldn't be unisoned in the WC folder so lets deal with this later. besides, its mainly being used only for HEP right now.
# if ismac():
# very interesting, and it worked, but overly complicated... why not just use an Wolfram APIFun as a more formal server?
# also, don't need this right now until I have a VM again.
# checkWL()