FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
