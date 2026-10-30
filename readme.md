# Doc4LLM / DocChecker

This repository contains the source code and experimental materials for **Detecting Contradictions Within and Across Documents**. The project implements **DocChecker**, a framework for detecting contradictions within a document and across multiple documents by combining document graph construction, graph enrichment, RICO rule discovery, and rule-based contradiction checking.

The repository is released primarily as an **available artifact**. It allows readers to inspect the implementation, datasets used by the paper, and experimental organization. It is organized as research code: different datasets and experiments use separate scripts, and there is currently no single command that runs the entire project.

## Repository structure

```text
code-for-Doc4LLM/
├── datasets/
├── docker/
├── doc_to_triplets/
├── DocChecker_and_exps/
├── requirements.txt
└── Detecting Contradictions Within and Across Documents (full version).pdf
```

### `datasets/`

Contains information about the 15 datasets used in the paper. The datasets can be obtained from their original publications or repositories such as Hugging Face. Please follow the licenses and terms of the original dataset providers.

See `datasets/readme.txt` for the complete dataset list and corresponding references.

### `docker/`

Contains the Dockerfile, detailed container instructions, and the exported `docchecker-sigmod26-ari.tar` image archive when it has been built. See the Docker quick start below or `docker/README.md` for build, load, and execution commands.

### `doc_to_triplets/`

Contains the basic document-to-graph implementation. Its main purpose is to extract entities and relations from documents and represent them as triples or RDF-style graphs.

Representative code includes:

- `basic_model.py`: basic document-to-triple processing.
- `generate_with_llm_v3.py`: uses an LLM to extract entities and relations from documents.
- `generate_attribute_graph.py`: constructs graphs containing entity, relation, and attribute information.
- `attribute_graph_enrich/`: enriches graph nodes with additional attributes.
- `attribute_graph_join/`: joins related graph structures.
- `RICO_predicates/`: defines predicates used by RICO rules.
- `RICO_rule_discovery/`: discovers candidate contradiction-detection rules from graph patterns.

This directory is useful when only document-to-triple conversion is needed, without running the complete DocChecker workflow.

### `DocChecker_and_exps/`

Contains the main DocChecker implementation and the experimental code used in the project.

- `DocChecker/`: dataset-specific DocChecker pipelines for the datasets used in the paper.
- `rule_discovery_intra_doc/`: rule discovery for contradictions within one document.
- `rule_discovery_inter_doc/`: rule discovery for contradictions across documents.
- `rule_predicates/`: predicate implementations for entities, relations, temporal information, fuzzy language, NLI, and other contradiction types.
- `graph_enrich/`: graph enrichment utilities.
- `graph_join/`: graph joining utilities.
- `experiments/`: experimental, ablation, and scalability code.
- `downstream_tasks/`: exploratory applications in NLP, machine learning, and retrieval-augmented generation.

The `DocChecker/` directory contains implementations for datasets including ContractNLI, ContraDoc, DocNLI, MNLI, SNLI, CaseHOLD, ConTRoL, ECON, StorySumm, RTE, and other datasets described in the paper.

## Main processing workflow

DocChecker generally follows this pipeline:

```text
Source documents
      ↓
Dataset extraction and preprocessing
      ↓
Entity-relation extraction with an LLM
      ↓
Attribute graph construction and enrichment
      ↓
Graph pattern and RICO rule discovery
      ↓
Contradiction detection and evaluation
```

The exact filenames and scripts differ between datasets, but the main stages are similar:

1. Convert a source dataset into the JSON format expected by the selected dataset pipeline.
2. Extract entities, relations, and text spans from each document.
3. Filter and enrich the resulting graph.
4. Discover star or dual-star graph patterns.
5. Generate RICO predicates and contradiction rules.
6. Apply or evaluate the generated rules on the target dataset.

## Docker quick start

The recommended way to run the artifact is the provided Docker image. It includes the source code and intermediate artifacts already present in the repository, with the CPU build of PyTorch 2.7.1. No NVIDIA GPU is required.

Docker image (1,519,215,104 bytes):

<https://github.com/chen760316/Doc4LLM/releases/download/sigmod26-ari/docchecker-sigmod26-ari.tar>

From the repository root, build the image:

```bash
docker build -f docker/Dockerfile -t docchecker:sigmod26-ari .
```

Or load the exported image from `docker/docchecker-sigmod26-ari.tar` when the archive is available:

```bash
docker load -i docker/docchecker-sigmod26-ari.tar
```

Start an interactive container:

```bash
docker run --rm -it docchecker:sigmod26-ari
```

The project is located at `/workspace` inside the container. Pass LLM credentials at runtime, for example with `-e OPENAI_API_KEY`; never bake them into the image. To keep generated files after the container exits, bind-mount the host checkout at `/workspace`.

For output persistence, an offline ContractNLI smoke workflow, and image export instructions, see `docker/README.md`.

## Native environment setup

### 1. Requirements

The project is implemented in Python. Python 3.11 is recommended for consistency with the Docker build. Some experimental scripts use PyTorch, TensorFlow, CUDA, or external LLM services; the required hardware depends on the selected experiment.

The supplied `requirements.txt` contains the pinned runtime dependencies used by `DocChecker_and_exps/` and `doc_to_triplets/`. Native GPU setup remains platform-specific; use the appropriate PyTorch installation source if the default package index does not provide the desired CUDA build.

### 2. Create a virtual environment

Open PowerShell in the repository root:

```powershell
cd E:\code-for-Doc4LLM
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

NLTK-based predicate scripts download the `punkt` tokenizer data on first use. The Docker image preinstalls both `punkt` and `punkt_tab`. Pretrained Hugging Face models and external datasets must be obtained separately when required by the selected experiment.

### 4. Configure the LLM API

Several scripts use an OpenAI-compatible client. Set the API key in the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
```

Alternatively, create a local `.env` file in the working directory:

```text
OPENAI_API_KEY=YOUR_API_KEY
```

Do not upload API keys to a public repository or include them in an artifact archive.

The model name is configured inside the corresponding Python scripts. Check that the configured model is available to your account before execution. LLM-based processing may incur API charges; reducing the sample size and worker count is recommended for a first run.

### 5. Prepare datasets

Obtain the desired dataset from its official source and place or convert it according to the selected dataset directory. Because the original datasets use different formats, each dataset has its own extraction script, commonly named `extract.py` or `extract_data.py`.

For example, the ContractNLI workflow is located at:

```text
DocChecker_and_exps/DocChecker/ContractNLI/
```

Its `extract.py` expects a CSV file named `contractnli_a_train.csv` with the columns `premise`, `hypothesis`, and `label`, and produces `sampled_data.json`.

## Example configuration and execution

The following ContractNLI example shows the normal execution order. Run these commands from the ContractNLI directory:

```powershell
cd E:\code-for-Doc4LLM\DocChecker_and_exps\DocChecker\ContractNLI
python extract.py
python generate_with_llm_v3.py
python filter.py
python graph_enrich.py
```

The scripts correspond to these stages:

| Script | Purpose | Main output |
| --- | --- | --- |
| `extract.py` | Samples and converts the source dataset | `sampled_data.json` |
| `generate_with_llm_v3.py` | Extracts an entity-relation graph with an LLM | `output_knowledge_graph_v3.json` |
| `filter.py` | Removes low-information graph edges | `filtered_knowledge_graph.json` |
| `graph_enrich.py` | Enriches frequent graph entities with Wikidata attributes | enriched graph JSON |

Before running `graph_enrich.py`, check its `OUTPUT_FILE` setting. The downstream scripts expect the enriched graph to be named `filtered_knowledge_graph_enrich.json`; some versions of the script currently use `test_async.json` as the default output name.

To continue with graph pattern and rule discovery:

```powershell
cd dual_star_output_multi_hop
python dual_star_discovery_multi_hop.py
python attrs_add.py
python rule_generation_llm.py
```

These scripts discover multi-hop dual-star graph structures, attach graph attributes, and generate candidate RICO predicates and rules. Generated files are written into subdirectories such as `dual_star_json/`, `dual_star_json_with_attrs/`, and `dual_star/llm_rules/`.

Other dataset directories use a similar organization, but input paths and label mappings may differ. Review the configuration constants near the beginning or end of the selected script before running it.

## Experiments

The main experimental directories are:

- `experiments/exp-1/`: contradiction-detection and dense-document model experiments.
- `experiments/exp-2/`: evaluation code organized by research question.
- `experiments/exp-ablation/`: ablation studies, including variants without graph enrichment or graph joining.
- `experiments/exp-scalibility/`: scalability experiments that vary dataset size and other parameters.

Experimental scripts may require pretrained models, generated intermediate files, or dataset-specific inputs that are not downloaded automatically. Check the input and output path constants in the selected script before execution.

## Notes for artifact users

- Run scripts from their own dataset or experiment directory because many paths are relative.
- Preserve included intermediate JSON files before rerunning scripts, since fixed output filenames may overwrite them.
- LLM output may vary even when the dataset sampling seed is fixed.
- Wikidata enrichment requires an internet connection and may be affected by API rate limits.
- The repository provides source code and selected artifacts, but does not currently provide a fully automated one-command reproduction workflow.
- Dataset and model licenses remain governed by their original providers.

## Paper

`Detecting Contradictions Within and Across Documents (full version).pdf` is the extended version of the paper and contains the complete technical description referenced by the submitted version.

When citing this artifact, use the final publication metadata for **Detecting Contradictions Within and Across Documents** once it is available.
