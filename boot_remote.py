if __name__ == '__main__':
    import boot
    boot.take_tic_from_sysargv()
    import pickle
    with open('_remote_scope.pkl', 'rb') as scope:
        globals().update(pickle.load(scope))
    with open('_remote_script.py', 'r') as pyscript:
        exec(pyscript.read())
    boot.finish_dnn_remote()
