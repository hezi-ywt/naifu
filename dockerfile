# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.08-py3

# Set the working directory in the container
WORKDIR /workspace/naifu/

# Copy the current directory contents into the container at /workspace/naifu
COPY . /workspace/naifu/

# Configure environment variables for proxies (if required)
# Replace the below proxy URLs with your actual proxy server's URLs
ENV http_proxy=http://214.28.51.55:13128
ENV https_proxy=http://214.28.51.55:13128

# Install the python dependencies specified in requirements.txt
RUN pip install -r requirements.txt
# If you do not need proxy to install dependencies, use:
# RUN pip install -r requirements.txt

# Default command to run the script
# CMD ["python", "script.py"]


# CMD ["sh", "-c", "PYTHONPATH=./ sh ./hydit/train.sh --index-file dataset/porcelain/jsons/porcelain_mt.json --multireso --reso-step 64 "]