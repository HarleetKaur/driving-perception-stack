
FROM ros:humble

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --default-timeout=600 --retries 10 "numpy<2" torch torchvision

RUN pip3 install --no-cache-dir --default-timeout=600 --retries 10 ultralytics lapx

WORKDIR /ws
