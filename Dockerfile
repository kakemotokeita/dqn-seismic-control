FROM python:3.8

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip

ADD . /usr/src/app/
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

ENTRYPOINT bash
