# Legal NLP Capstone

A research project exploring language modeling and natural language processing for legal documents. Across seven phases, the notebooks develop and evaluate legal-domain models for text generation, entity recognition, rhetorical-role labeling, judgment prediction, statute identification, summarization, and prior-case retrieval.

The project compares four model families—**Legal Transformer**, **Legal-GraphProp**, **encoder-decoder** and **Legal-CapsuleRoute**—and experiments with transfer learning, parameter-efficient fine-tuning, and knowledge distillation.

> **Project format:** This repository is a collection of Jupyter notebooks, not a single installable application. Notebooks may generate supporting Python modules during execution. There is no repository-wide dependency lockfile or command that runs every experiment.

## Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Methodology](#methodology)
- [Phase Guide](#phase-guide)
- [Repository Structure](#repository-structure)
- [Data and Model Requirements](#data-and-model-requirements)
- [Execution Guide](#execution-guide)
- [Outputs and Reproducibility](#outputs-and-reproducibility)
- [Limitations](#limitations)

## Project Overview

The experiments follow a progression from legal-domain language modeling to task-specific applications:

1. Develop legal language-model backbones.
2. Adapt pretrained backbones to legal NLP tasks.
3. Compare architectures and fine-tuning strategies.
4. Evaluate on task-specific development data.
5. Produce predictions, metrics, and submission artifacts.

Each phase contains independent experiments. The data format, model requirements, configuration, evaluation procedure, and output files may vary between notebooks.

## Architecture

### Model families

**Legal Transformer**  
A decoder-only Transformer for legal text. The notebooks implement causal self-attention, rotary position embeddings (RoPE), GELU feed-forward layers, and tied token/output embeddings. The Phase 1 Legal Transformer notebook specifies a 896-dimensional, 27-layer, 14-head configuration and trains a byte-level BPE tokenizer.

**Legal-GraphProp**  
A sequence model using local graph-style message passing in place of full self-attention. Its `GraphPropLayer` scores connections between nearby token positions, aggregates projected messages using relative-position embeddings, and applies a residual update. GraphProp is explored both as a language model and as a downstream task backbone.

**Legal-CapsuleRoute**  
A model that routes information between nearby token representations using capsule-style routing by agreement. Its `CapsuleRoutingLayer` computes votes and iteratively updates routing weights, with feed-forward layers between routing layers.

**encoder-decoder**
a traditional encoder decoder architecture in which encoder generates contextualized representation and decoder use it to generate text

### Common downstream pipeline

Most task notebooks use a variation of this process:

1. Configure the experiment, runtime, artifact paths, and output directory.
2. Locate input files or download the task dataset.
3. Load a matching checkpoint and tokenizer when the notebook uses a pretrained backbone.
4. Normalize records and tokenize text.
5. Prepare task examples, such as document chunks, sentence sequences, or query/candidate pairs.
6. Train a task head or fine-tune the backbone.
7. Evaluate and select settings using development data.
8. Generate test predictions and save metrics and artifacts.

### Task-specific components

- **Long-document classification:** Cases are split into chunks, encoded, and pooled into document representations. Some pipelines use position features, a bidirectional GRU, and learned attention to combine chunk vectors.
- **NER:** Character-offset spans are aligned to words and represented with B/I/O tags. Token offsets connect the tokenizer output to the original text; the notebooks use overlapping windows for long documents. Viterbi decoding is available to enforce valid tag transitions.
- **Rhetorical-role labeling:** Sentence representations are classified into rhetorical-role categories, with sentence-context modeling available in the pipeline.
- **Summarization:** A sentence selector identifies useful source sentences; a generative model rewrites selected content, and the pipeline assembles candidate sentences under a word budget.
- **Prior-case retrieval:** Query and candidate representations are scored using cosine similarity. Neural scores are combined with lexical BM25-like and TF-IDF scores, with development-set tuning for ranking and post-processing.

### Fine-tuning and training techniques

The notebooks experiment with full and partial fine-tuning, LoRA, rsLoRA, LoRMA, bottleneck adapters, and frozen-backbone task heads. Depending on the experiment, training also uses mixed precision, gradient accumulation, gradient clipping, checkpointing, class weighting, and teacher-student distillation.

## Methodology

The project uses staged, validation-driven experiments:

1. **Pretraining:** Phase 1 trains legal language models from scratch using legal text and byte-level BPE tokenization. Autoregressive next-token training is used for language-modeling experiments.
2. **Task adaptation:** Later phases add task-specific preprocessing and prediction heads to pretrained backbones. The notebooks compare alternative architectures and fine-tuning approaches.
3. **Development evaluation:** Development or hold-out data is used for model selection, threshold tuning, decoding choices, sentence selection, or retrieval-score blending.
4. **Long-text processing:** Long judgments are handled with chunks or windows instead of being passed as a single unrestricted sequence.
5. **Distillation and imbalance handling:** Some experiments use teacher scores, class or positive-label weights, rare-label handling, or probability adjustment.
6. **Test inference and packaging:** Selected configurations generate predictions and save summaries, score files, plots, and submission ZIPs.

## Phase Guide

### Phase 1 — Legal language-model pretraining

Notebooks explore language-model pretraining and generation with several architectures:

- `capsuleroute.ipynb` — CapsuleRoute language model.
- `graphporpagation.ipynb` — GraphProp language model.
- `legal-transformer.ipynb` — decoder-only Legal Transformer.
- `encoder-decoder-phase1 (2).ipynb` — encoder-decoder experiment.

The language-model notebooks load legal text, train or load a byte-level BPE tokenizer, prepare training and validation data, train a model, evaluate generated continuations, and save checkpoints and predictions. Notebook settings differ; consult the selected notebook for its exact data caps, model dimensions, training schedule, and generation configuration.

### Phase 2 — Legal named entity recognition

Notebooks include:

- `lorma_legal_transformer.ipynb`
- `adaptive-graphprop-p2.ipynb`
- `p2-encdec (3).ipynb`
- `p2-kd-lorma-capsule.ipynb`

The task identifies legal entities in judgments. The pipeline aligns entity spans to words, predicts B/I/O tags, evaluates strict span-level metrics, and produces test predictions. Experiments include Legal Transformer with LoRMA, GraphProp with adapters or partial fine-tuning, and additional encoder-decoder and distillation variants.

### Phase 3 — Rhetorical-role labeling

Notebooks include:

- `capsule-route-rslora.ipynb`
- `graphporp-lora-p3 (1).ipynb`
- `lorma-transformer-latest.ipynb`
- `p3-kd_transformer_encdec (1).ipynb`

The task assigns one rhetorical-role label to each sentence in legal documents. The notebooks include a CapsuleRoute rsLoRA experiment and other backbone/adaptation variants. The generated `rr_core.py` pipeline handles data, sentence encoding, training, evaluation, and prediction; `rr_peft.py` defines parameter-efficient fine-tuning helpers.

### Phase 4 — Court judgment prediction

Notebooks include:

- `KD_transformer_to_capsule.ipynb`
- `p4-encdec_fulfit.ipynb`
- `p4-finetune-legal-transformer.ipynb`
- `p4-grapprop.ipynb`

The task predicts a binary judgment outcome from a case. In the main Legal Transformer notebook, `cj_core.py` loads and chunks documents, encodes chunks, aggregates them with a document-level head, evaluates development performance, tunes a decision threshold, and writes predictions. `cj_peft.py` contains fine-tuning and memory-management helpers.

### Phase 5 — Legal statute identification

Notebooks include:

- `kd-p5.ipynb`
- `lorma-p5-updated.ipynb`
- `p5-encdec (1).ipynb`
- `p5-graphprop.ipynb`

The task predicts one or more applicable statutes for each case. The main pipeline, written as `lsi_core.py`, normalizes records, chunks documents, creates document-level representations, and trains a multi-label classifier. Experiments include teacher-logit distillation into a CapsuleRoute student, threshold tuning, and class-imbalance handling.

### Phase 6 — Abstractive legal summarization

Notebooks include:

- `p6-encdec.ipynb`
- `p6-graphprop-adapters.ipynb`
- `p6-legal-lorma-transformer.ipynb`
- `p6_KD.ipynb`

The summarization experiments select important sentences from long judgments and may rewrite them with a generative model. The primary Legal Transformer notebook writes `p6_core.py` and model code during execution. The pipeline includes sentence splitting, feature construction, selector training, source-summary pair preparation, fine-tuning, summary assembly, development evaluation, and submission packaging.

### Phase 7 — Prior-case retrieval

Notebooks include:

- `kd-p7.ipynb`
- `p7-enc-dec.ipynb`
- `p7-graphprop.ipynb`
- `p7-lorma-finetune-legal-transformer.ipynb`

The task ranks candidate prior cases for each query case and evaluates the top seven results using Micro-F1@7. The notebooks compute lexical BM25-like and TF-IDF scores, neural document-similarity scores, and development-tuned score combinations. The LoRMA Legal Transformer notebook can generate score matrices for use as a teacher; the CapsuleRoute KD notebook expects those scores when configured for distillation.

## Repository Structure

```text
Capstone_project/
├── Phase-1/    Legal language-model pretraining and generation
├── Phase-2/    Legal named entity recognition
├── Phase-3/    Rhetorical-role labeling
├── Phase-4/    Court judgment prediction
├── Phase-5/    Legal statute identification
├── Phase-6/    Abstractive legal summarization
├── Phase-7/    Prior-case retrieval
└── README.md   Project documentation
```

The notebooks are the primary source files. Some notebooks use `%%writefile` to create modules such as `arch_legal_transformer.py`, `arch_capsuleroute.py`, `p6_core.py`, or `lsi_core.py` in the notebook's working directory. These generated modules may not exist in the repository before the notebook is run.

## Data and Model Requirements

### Dataset

Several notebooks refer to the Hugging Face dataset:

- `Exploration-Lab/CS-781-Capstone`

Some data may require authentication or acceptance of dataset terms. Where required, configure a Hugging Face read token as `HF_TOKEN` using the notebook platform's secret manager or an environment variable. **Do not commit tokens or other secrets to the repository.**

The exact files and schemas vary by phase. Notebooks may expect JSON files for train, development, and test splits, label vocabularies, entity spans, or candidate cases.

### Checkpoints and tokenizers

Downstream experiments commonly need a pretrained model checkpoint (`.pt`) and its matching tokenizer (`.json`). Attach or provide the correct artifacts for the selected architecture and notebook. A checkpoint and tokenizer from different model runs or model families may be incompatible.

Some experiments also depend on outputs from other notebooks—for example, teacher logits or Phase 7 retrieval score matrices. These artifacts are not necessarily included in the repository.

### Runtime and dependencies

A GPU-backed Kaggle or Colab environment is recommended for full training. CPU execution may be possible for some steps, but large models and long-document experiments can be slow or memory-intensive.

There is no checked-in `requirements.txt`, `pyproject.toml`, or environment lockfile. Depending on the notebook, dependencies include:

- PyTorch
- NumPy
- scikit-learn
- SciPy
- Hugging Face `datasets` and `huggingface_hub`
- `requests` and `ijson`
- `tqdm`
- Matplotlib
- BERTScore

A general-purpose notebook environment can be prepared with:

```bash
python -m pip install jupyterlab numpy scikit-learn scipy \
  datasets huggingface_hub requests ijson tqdm matplotlib bert-score
```

Install PyTorch separately if it is not already provided by your runtime. Choose a build compatible with the installed CUDA version.

## Execution Guide

### Run a notebook experiment
### phase 1 notebooks just need access to datasets to run them
### from phase 2 to 7 notebooks required .pt files and tokenizer files of pretrained models

1. Clone the repository and open the notebook for the task and model you want to run.
2. Read the notebook's introductory markdown and configuration cells.
3. Select a suitable runtime, preferably with a GPU for training.
4. Configure Hugging Face access if the notebook needs gated data.
5. Attach the required dataset files, checkpoint, tokenizer, and any teacher artifacts.
6. Review and update paths such as `CKPT`, `TOK`, `OUT`, or data directories. Some notebooks use Kaggle-specific paths such as `/kaggle/input` and `/kaggle/working`.
7. Run notebook cells **in order**. Cells using `%%writefile` generate helper modules used by later cells.
8. Review validation metrics and saved summaries.
9. Download the required checkpoints and output files before a hosted session expires.

For local Jupyter, a starting point is:

```bash
git clone https://github.com/preeetpatel59-sketch/Capstone_project.git
cd Capstone_project
python -m pip install jupyterlab numpy scikit-learn scipy \
  datasets huggingface_hub requests ijson tqdm matplotlib bert-score
jupyter lab
```

This prepares a general environment, but does not install a CUDA-specific PyTorch build or provide dataset access, checkpoints, tokenizers, or teacher artifacts.

### Suggested order for dependent experiments
### for distillation experiment , first run the teacher and attach the output zip while running distillation experiment 
Most notebooks can be run independently when their required inputs are available. For chained experiments:

- **Backbone to downstream task:** run or obtain the relevant Phase 1 model and tokenizer, then attach both to the downstream notebook using that backbone.
- **Phase 5 distillation:** produce teacher logits first, then attach them to a student experiment configured to use them.
- **Phase 6 distillation:** provide the teacher outputs or training pairs required by the selected notebook configuration.
- **Phase 7 distillation:** run `p7-lorma-finetune-legal-transformer.ipynb` first to create the teacher score artifact expected by `kd-p7.ipynb`, then attach that artifact to the student run.

Always check the selected notebook's configuration and introductory cells for the exact artifact filenames and paths.

### Resource and resume guidance

Some notebooks provide smoke-test settings, resumable checkpoints, or automatic batch-size adjustments. Use a smoke test before committing to a full run when available. If you encounter memory limits, reduce batch size, chunk length, or other resource-intensive settings; use gradient accumulation where the notebook supports it. Full experiments may require hours and may be affected by hosted-session limits.

## Outputs and Reproducibility

Depending on the experiment, notebooks may produce:

- Model checkpoints (`.pt`)
- Tokenizer files (`.json`)
- Training histories and experiment summaries (`.json`)
- Score or probability matrices (`.npz`)
- Prediction JSON files
- Submission ZIP files, commonly containing `prediction.json`
- Plots, including loss curves and confusion matrices
- Aggregate archives such as `*_ALL_outputs.zip`

For reproducible comparisons, retain the notebook, its configuration, matching checkpoint and tokenizer, data version, random seed, and output summary together. Results may vary with hardware, library versions, random seeds, data availability, and stochastic generation.

## Limitations

- This repository contains research and coursework experiments; it is not a deployed service or a legal-advice system.
- Model-generated text and predictions should not be treated as legal advice or relied on without qualified review.
- Data access, dataset terms, hosted runtimes, and some model artifacts are external to this repository.
- Notebooks have different preprocessing, configuration, evaluation methods, and output schemas.
- Many notebooks assume Kaggle-style directory paths; update paths for other environments.
- There is no repository-wide test suite or single command for executing all phases.
- Generated helper modules may only appear after running their notebook cells.
