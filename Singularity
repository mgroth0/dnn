Bootstrap: docker
From: ubuntu:20.04

%runscript
    echo "The runscript is the containers default runtime command!"
    # home is "/root" and this is the cwd
    cd dnn
    ./dnn
    exec echo "exec in the runscript replaces the current process!"

%labels
   AUTHOR mjgroth@mit.edu

%post
    echo "The post section is where you can install, and configure your container."
    apt update
    apt full-upgrade -y
    apt autoremove
    apt install curl -y
    curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x ~/miniconda.sh
    ~/miniconda.sh -b
    ~/miniconda3/bin/conda create -y --name dnn python=3.8
    apt install git -y
    git clone https://github.com/mgroth0/dnn
    git clone https://github.com/mgroth0/mlib
    ~/miniconda3/bin/conda init
    source .bashrc
    conda activate dnn
    #cd dnn
    #conda install --file requirements.txt
    echo "done with post-build"