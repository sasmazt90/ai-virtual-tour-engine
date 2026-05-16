FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS opensplat-build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    libopencv-dev \
    ca-certificates \
    wget \
    unzip \
  && rm -rf /var/lib/apt/lists/*

ARG TORCH_VERSION=2.2.1
ARG CUDA_SHORT=121
ARG CMAKE_CUDA_ARCHITECTURES="75;80;86;89"

WORKDIR /opt
RUN git clone --depth 1 https://github.com/pierotofy/OpenSplat OpenSplat \
  && wget --no-check-certificate -nv \
    "https://download.pytorch.org/libtorch/cu${CUDA_SHORT}/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu${CUDA_SHORT}.zip" \
    -O libtorch.zip \
  && unzip -q libtorch.zip -d OpenSplat \
  && rm libtorch.zip \
  && cmake -S OpenSplat -B OpenSplat/build -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/OpenSplat/libtorch \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
  && cmake --build OpenSplat/build --config Release

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    OPEN_SPLAT_BIN=/usr/local/bin/opensplat \
    LD_LIBRARY_PATH=/opt/libtorch/lib:${LD_LIBRARY_PATH}

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    colmap \
    ca-certificates \
    libopencv-dev \
    libgomp1 \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY worker.py .
COPY --from=opensplat-build /opt/OpenSplat/build/opensplat /usr/local/bin/opensplat
COPY --from=opensplat-build /opt/OpenSplat/libtorch /opt/libtorch

CMD ["python3", "worker.py"]
