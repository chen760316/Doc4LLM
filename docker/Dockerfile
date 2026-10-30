FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="DocChecker" \
      org.opencontainers.image.description="SIGMOD 2026 ARI artifact for Detecting Contradictions Within and Across Documents"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    HF_HOME=/home/artifact/.cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt \
    && python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab

RUN useradd --create-home artifact \
    && mkdir -p /home/artifact/.cache/huggingface \
    && chown -R artifact:artifact /home/artifact /workspace

COPY --chown=artifact:artifact DocChecker_and_exps ./DocChecker_and_exps
COPY --chown=artifact:artifact doc_to_triplets ./doc_to_triplets
COPY --chown=artifact:artifact datasets ./datasets
COPY --chown=artifact:artifact README.md requirements.txt ./
COPY --chown=artifact:artifact ["Detecting Contradictions Within and Across Documents (full version).pdf", "./"]

USER artifact

CMD ["/bin/bash"]
