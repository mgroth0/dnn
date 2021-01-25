Bootstrap: docker
From: ubuntu:20.04

%runscript
    echo "The runscript is the containers default runtime command!"
    # home is "/root" and this is the cwd

    # these are now bound and pulled from outside
    #cd /matt/mlib
    #git reset --hard #undo chmod so can pull
    #git pull
    #cd /matt/dnn
    #git reset --hard #undo chmod so can pull
    #git pull


    cd /matt

    # Not sure why I felt te need to do this each time. They are hardly ever updated.
    #echo copy1
    #cp -r _resources/_weights dnn
    #echo copy2
    #cp -r _ImageNetTesting dnn
    #echo copy3
    #cp -r _data dnn

    cd dnn

    env
    # bash
    # CONDA_HOME=/matt/miniconda3

    echo "CONDA_HOME in singularity runscript is:"$CONDA_HOME
    ./dnn "$@"
    #exec bash
    exec echo "exec in the runscript replaces the current process!"

%labels
   AUTHOR mjgroth@mit.edu

%post
    echo "The post section is where you can install, and configure your container."
   
    echo "done with post-build"