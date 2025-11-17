# syntax=docker/dockerfile:1

# Build argument to use remote base image or build locally
ARG USE_REMOTE_BASE=false
ARG BASE_IMAGE=us-east1-docker.pkg.dev/anita-hunter/mark-ip3g-repo/ip3g:base

# Base stage: dependencies (push this rarely)
FROM continuumio/miniconda3:latest AS base

# Set up environment variables for non-interactive installs and headless matplotlib
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# System packages for scientific Python stacks (optional, most are included in conda-forge)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy environment file and create conda environment
COPY environment.yml /tmp/environment.yml
ARG CONDA_ENV_NAME=dataexplr
ENV CONDA_ENV_NAME=$CONDA_ENV_NAME
ENV PATH=/opt/conda/envs/$CONDA_ENV_NAME/bin:$PATH
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# Activate conda env by default in all future RUN/CMD
SHELL ["/bin/bash", "-lc"]

# Remote base stage: pull pre-built base from registry
FROM ${BASE_IMAGE} AS base-remote

# Runtime stage: add application code
FROM base${USE_REMOTE_BASE:+-remote} AS runtime

WORKDIR /app

# Declare data/output volumes so they can be mounted from host
VOLUME ["/app/loaded_data", "/app/images"]

# Copy Python code last so changes don't invalidate dependency layers
COPY *.py /app/
COPY requirements.txt /app/

# Default command runs the visualization pipeline; override with `docker run ... python <script>.py`
ENTRYPOINT ["python", "-u"]
CMD ["visualize_data.py"]
