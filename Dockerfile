FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LABEL maintainer="yoonhyek@smilegate.com"

RUN apt-get update && apt-get install -y \
    libpq-dev \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace
WORKDIR /workspace

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir transformers[sentencepiece]

CMD ["/bin/bash"]

