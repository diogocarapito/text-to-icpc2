FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/hf
WORKDIR /app

# CPU-only torch first (keeps the image ~1 GB smaller and cached across code changes)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY data/raw/icpc2_ptpt.csv data/raw/icpc2_en_titles.csv data/raw/
RUN pip install -e ".[serve]"

# Commit this image was built from (no .git in the image); the GPU loop clones it. Set by deploy.sh.
ARG GIT_SHA=""
ENV ICPC2_GIT_SHA=$GIT_SHA

RUN useradd -m app && mkdir -p /hf /app/data && chown app /hf /app/data
USER app
EXPOSE 8000
CMD ["uvicorn", "icpc2.web.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
