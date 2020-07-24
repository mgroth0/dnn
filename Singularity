Bootstrap: docker
From: ubuntu:20.04

%runscript
    echo "The runscript is the containers default runtime command!"
    # home is "/root" and this is the cwd
    cd /matt/mlib
    git pull
    cd /matt/dnn
    git pull
    env
    # bash
    # CONDA_HOME=/matt/miniconda3

    echo "CONDA_HOME in singularity runscript is:"$CONDA_HOME
    ./dnn
    #exec bash
    exec echo "exec in the runscript replaces the current process!"

%labels
   AUTHOR mjgroth@mit.edu

%post
    echo "The post section is where you can install, and configure your container."
    apt update
    apt full-upgrade -y
    apt autoremove
    apt install curl -y
    apt install libgl1-mesa-glx -y #https://github.com/conda-forge/pygridgen-feedstock/issues/10
    cd /
    mkdir matt
    cd matt
    curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p miniconda3
    apt install git -y
    git clone https://github.com/mgroth0/dnn
    git clone https://github.com/mgroth0/mlib
    /matt/miniconda3/bin/conda create -y --name dnn python=3.8
    /matt/miniconda3/bin/conda config --add channels conda-forge
    /matt/miniconda3/bin/conda config --add channels mgroth0
    cd dnn
    /matt/miniconda3/bin/conda install -y -n dnn --file=requirements.txt
    /matt/miniconda3/bin/conda install -y -n dnn tensorflow-gpu=2.2.0
    apt install iputils-ping -y
    /matt/miniconda3/bin/conda install gdown -y
    cd ..
    /matt/miniconda3/bin/gdown "https://drive.google.com/uc?id=1wauVN6nG3tKv7VifIfRVBL0fj8XfefVa"
    apt install unzip -ysanity
    unzip _resources.zip
    mv _resources/_weights dnn
    echo "done with post-build"