# Projector timing benchmark image.
# Build from the ct_laboratory repo root:
#   docker build -f benchmark/Dockerfile -t 172.20.157.23:5000/ct-lab-benchmark:latest .
#
# Official PyTorch image (no local torch build). CUDA 13.0 is the max the
# axis03 / gpu1 drivers (580.x) support; gpu2/5 (610.x) are newer and fine.
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel

# 7.5=T4/RTX20xx 8.0=A100 8.6=A40/3090 8.9=4090/L40S 9.0=H100/H200
# 10.0=B200 12.0=RTX5090; +PTX on the last for future arches (CUDA 13 min is 7.5)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda

RUN pip install --no-cache-dir --break-system-packages ninja tqdm

# Build and install ct_laboratory (compiles the CUDA extension with nvcc)
WORKDIR /opt/ct_laboratory
COPY setup.py ./
COPY src ./src
COPY ct_laboratory ./ct_laboratory
RUN pip install --no-build-isolation --no-cache-dir --break-system-packages .

# openssh-server so RunPod's SSH proxy/direct SSH can reach the pod
# (sshd is NOT started by default; cloud entrypoints start it explicitly)
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd

# Benchmark script + bundled 240-view geometry (~70 MB)
COPY benchmark/projector_timing_benchmark.py /opt/benchmark/
COPY benchmark/geometry /opt/benchmark/geometry

WORKDIR /opt/benchmark
ENTRYPOINT ["python", "projector_timing_benchmark.py"]
# cuda kernels at all power-of-two resolutions; pure-torch backends only at 64
CMD ["--resolutions", "64", "128", "256", "512", "--backends", "cuda", "torch-gpu", "torch-cpu"]
