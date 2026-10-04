FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    build-essential \
    ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/pip3.11 /usr/bin/pip

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"

RUN pip install -r requirements.txt

WORKDIR /root/shared/workspace