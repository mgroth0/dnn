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
    rm miniconda.sh
    apt install git -y
    git clone https://github.com/mgroth0/dnn
    git clone https://github.com/mgroth0/mlib
    /matt/miniconda3/bin/conda update -n base -c defaults conda
    /matt/miniconda3/bin/conda create -y --name dnn python=3.8
    /matt/miniconda3/bin/conda config --add channels conda-forge
    /matt/miniconda3/bin/conda config --add channels mgroth0
    cd dnn

    # I think trying to do it all at once is causing problems
    # /matt/miniconda3/bin/conda install -y -n dnn --file=requirements.txt
    while read p; do
      echo "installing conda package: $p"
      /matt/miniconda3/bin/conda install -y -n dnn $p
      echo "installed conda package: $p"
    done <requirements.txt


    apt install graphviz -y # https://github.com/XifengGuo/CapsNet-Keras/issues/7
    /matt/miniconda3/bin/conda install -y -n dnn tensorflow-gpu=2.2.0

    # NOTE: Python 3.9 users will need to add '-c=conda-forge' for installation
    /matt/miniconda3/bin/conda install -y -n dnn pytorch  -c pytorch
    /matt/miniconda3/bin/conda install -y -n dnn torchvision -c pytorch
    /matt/miniconda3/bin/conda install -y -n dnn torchaudio -c pytorch

    # takes forever to install, and its a confusing package. try without it?
    # /matt/miniconda3/bin/conda install -y -n dnn cudatoolkit=11.0 -c pytorch

    apt install iputils-ping -y
    /matt/miniconda3/bin/conda install gdown -y

    apt install unzip -y
    apt install zip -y # used in pipeline
    cd ..

    /matt/miniconda3/bin/gdown "https://drive.google.com/uc?id=1wauVN6nG3tKv7VifIfRVBL0fj8XfefVa"
    unzip _resources.zip
    rm _resources.zip

    /matt/miniconda3/bin/gdown "https://drive.google.com/uc?id=1SWxt9USdj1wB9sPUpV2M26ZEtO5eliyt"
    unzip _ImageNetTesting.zip
    rm _ImageNetTesting.zip

    # going to try to generate new images at runtime, like I used to
    #/matt/miniconda3/bin/gdown "https://drive.google.com/uc?id=1HSjjIHeze-bWCycmQrSbeDDpvhmhGqhv"
    #unzip _images.zip
    #rm _images.zip


    /matt/miniconda3/bin/gdown "https://drive.google.com/uc?id=1PQ3gop_fmV_Sp_GBAAeS3Gr1_ITkbo5G"
    unzip _data.zip
    rm _data.zip

    rm -rf __MACOSX # not sure where this comes from

    # binding these instead so I can use --nv
    # chmod -R 777 /matt/dnn
    # chmod -R 777 /matt/mlib #for pulling
    rm -rf /matt/dnn
    rm -rf /matt/mlib


    # using --nv instead
    # apt purge nvidia-*
    #apt install software-properties-common -y
    #add-apt-repository ppa:graphics-drivers/ppa
    #apt install nvidia-driver-440 -y #440 is the one used by OpenMind as I write this (polestar uses 430.50)


    cd /matt
    git clone https://github.com/onnx/onnx-tensorflow
    cd onnx-tensorflow
    /matt/miniconda3/envs/dnn/bin/pip install -e .



    echo "done with post-build"