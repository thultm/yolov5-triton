FROM nvcr.io/nvidia/tritonserver:23.12-py3
# Install dependencies
RUN pip install opencv-python pillow && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Copy models directory to the container
COPY models /models

CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=explicit"]