# DocChecker Docker image

This directory contains the container configuration for the SIGMOD 2026 ARI submission of **Detecting Contradictions Within and Across Documents**.

The image is named `docchecker:sigmod26-ari`. It contains the project source code and the intermediate artifacts already included in this repository under `/workspace`. It uses Python 3.11 with the CPU build of PyTorch 2.7.1, so no NVIDIA GPU is required.

## Prerequisites

- Docker with Linux container support (Docker Desktop on Windows is suitable).
- Enough disk space for the PyTorch runtime, Python dependencies, and approximately 750 MB of source code and included artifacts.

External datasets and pretrained models are not redistributed in the image. LLM API keys and other credentials are never baked into the image.

## Build the image

Run this command from the repository root:

```bash
docker build -f docker/Dockerfile -t docchecker:sigmod26-ari .
```

The same command works in PowerShell. The build requires network access to Docker Hub, the Python package index, the official PyTorch CPU package index, and the NLTK data host.

## Load a previously exported image

If `docchecker-sigmod26-ari.tar` is present in this directory, build tools are not needed. From the repository root, verify the archive when its SHA-256 file is available, then load it:

```bash
sha256sum -c docker/docchecker-sigmod26-ari.tar.sha256
docker load -i docker/docchecker-sigmod26-ari.tar
```

To create a new distributable archive and checksum after building:

```bash
docker save -o docker/docchecker-sigmod26-ari.tar docchecker:sigmod26-ari
sha256sum docker/docchecker-sigmod26-ari.tar > docker/docchecker-sigmod26-ari.tar.sha256
```

## Start a container

Open an interactive shell:

```bash
docker run --rm -it docchecker:sigmod26-ari
```

The working directory is `/workspace`. Scripts should still be run from their own dataset or experiment directory because many input and output paths are relative. All included neural-model dependencies use CPU execution.

## Provide API credentials

Pass credentials at runtime. For example, after setting `OPENAI_API_KEY` in the host shell:

```bash
docker run --rm -it -e OPENAI_API_KEY docchecker:sigmod26-ari
```

For scripts that also use an OpenAI-compatible endpoint, add `-e OPENAI_BASE_URL`. Other scripts may use provider-specific variables such as `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, or `DASHSCOPE_API_KEY`.

Alternatively, create a local `.env` file outside Docker and pass it with `--env-file .env`. Do not commit or upload this file.

## Preserve outputs and model downloads

The container can write to its included working tree, but `--rm` deletes those changes when it exits. To persist outputs in the host checkout, bind-mount the repository. In PowerShell:

```powershell
docker run --rm -it -v "${PWD}:/workspace" docchecker:sigmod26-ari
```

In a Linux or macOS shell:

```bash
docker run --rm -it -v "$(pwd):/workspace" docchecker:sigmod26-ari
```

On Linux, ensure that the mounted directories are writable by the container user. To retain Hugging Face downloads separately, add a named volume:

```bash
-v docchecker-hf:/home/artifact/.cache/huggingface
```

## Offline smoke workflow

The included ContractNLI graph artifacts allow the graph-filtering and dual-star-discovery stages to be exercised without dataset downloads or LLM API calls:

```bash
docker run --rm docchecker:sigmod26-ari bash -lc '
set -e
python -m pip check
cd /workspace/DocChecker_and_exps/DocChecker/ContractNLI
python filter.py
cd dual_star_output_multi_hop
python dual_star_discovery_multi_hop.py
test -s ../filtered_knowledge_graph.json
test -d dual_star_json
'
```

This checks the environment and selected offline pipeline stages, not every dataset or experiment. Full reproduction still requires the source datasets, any generated inputs, pretrained models needed by the selected script, and external services described in the root `README.md`.

## Experiment notes

- Review path constants and model names near the beginning or end of each script before execution.
- Fixed output filenames may overwrite included intermediate artifacts. Use a bind mount only when host-side changes are intended, or copy the original data first.
- LLM output can vary, and API calls may incur charges.
- Wikidata, Wikipedia, Hugging Face, and LLM workflows require network access and may be rate-limited.
- Distributed scripts under `parallel/` contain site-specific host addresses and require separate SSH infrastructure; the image does not create those machines.
- TensorFlow is installed for the downstream TextCNN code; its Linux package uses CPU execution in this image.
