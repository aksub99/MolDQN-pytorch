FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y -q wget git libxrender1 libsm6 bzip2 && \
    apt-get clean

RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local && \
    conda install -q -y -c conda-forge rdkit

RUN apt-get -y update && \
    apt-get -y install cmake libopenmpi-dev python3-dev zlib1g-dev

ADD get-pip.py .

ADD requirements.txt .

RUN python get-pip.py

RUN pip install -r requirements.txt

RUN pip install tensorflow-gpu==1.14

RUN pip install torch==1.4.0

RUN pip install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"
