FROM ml_platform.yandex/user-default
ENV DEBIAN_FRONTEND noninteractive
# Write your instructions here

# DataSphere requires Python 3.7 and PIP
RUN\
 apt-get update &&\
 apt-get install -y --no-install-recommends apt-utils &&\
 apt-get install -y build-essential curl software-properties-common &&\
 add-apt-repository ppa:deadsnakes/ppa &&\
 apt-get update &&\
 apt-get install -y python3.7-dev python3-virtualenv &&\
 pip install torch==1.7.1 torchvision &&\
 curl 'https://bootstrap.pypa.io/get-pip.py' | python3.7 &&\
 wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip &&\
 unzip ninja-linux.zip -d /usr/local/bin/ &&\
 update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force