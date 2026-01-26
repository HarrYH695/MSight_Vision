FROM michigantrafficlab/msight-edge:v1.5.1-pytorch2.7.1-cuda12.8-cudnn9

# Install build tools and rsync
RUN apt-get update && \
    pip install --upgrade build

# Copy the original source to a temporary location
COPY . /tmp/msight-vision

# Set working directory
WORKDIR /tmp/msight-vision

# Install the obfuscated package
RUN pip install .

# Clean up temporary source code
RUN rm -rf /tmp/msight-vision
