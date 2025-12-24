# Automated Knowledge Component Generation for Dialogue Knowledge Tracing

Honors thesis project under Professor Andrew Lan. Directly expands on code and work from Alexander Scarlatos on Dialogue KT (see the paper <a href="https://arxiv.org/abs/2409.16490">Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs</a>).

## Description

Explores automated KC generation in comparison to using preexisting, manually curated KCs. Uses LLMs to automatically generate KCs for given dialogues, and refines these KCs using unsupervised clustering (HDBSCAN and agglomerative clustering) and GPT-4.1. Compares performance of generated KCs to Common Core standards (annotated using the Achieve The Core hierarchy). All dialogues are from the Eedi QATD2k dataset (<a href="https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k">Question Anchored Tutoring Dialogues</a>).

## Environment

Uses Python 3.10.12. Packages and versions used have been specified in both `requirements.txt` and `requirements.yml`. Note that small modifications to packages were made to resolve version compatability issues. Specifically, line 273 of `peft/src/peft/tuners/lora/bnb.py` was removed to remove outdated use of `memory_efficient_backward` property. In addition, some version compatability issues seemed to arise, requiring all instances of ``ensure_all_finite`` arguments to be removed when calling ``check_array`` in hdbscan. These issues should not be present if the package versions are installed correctly.

Environment variables were also set up via the following:
```
export OPENAI_API_KEY=<your key here>
export HF_TOKEN=<your token here>
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Data and Code
Data can be found in `eedi/train_gpt-4.1_updated.csv`, `eedi/val_gpt-4.1_updated.csv`, and `eedi/test_gpt-4.1_updated.csv`. Annotated data used for training can be found in `data/annotated`. For a full list of generated KCs, see `eedi/final_generated_kc_list.json`.

Most code (i.e. used to generate annotated data and figures) can be found in `eedi/process_data.ipynb`. Stored batch responses from the OpenAI API can be found in `eedi/batches`.

## Training
Models were trained using the dialogue kt framework. Run `train_eedi_atc.slurm` and `train_eed_gpt.slurm` to fine-tune Llama-3.1-8B-instruct on the Common Core and GPT-4.1 generated KCs, respectively.

Check the `results` folder for metric summaries and turn-level predictions for analysis. Output logs from training are generated in the `logs` folder. 

For the logs from the two models trained for this project, see `logs/lmkt_50653411.out` and `logs/lmkt_50657916.out`.

Saved model info is stored in `saved_models`. The actual weights and safetensor files were not uploaded.

Learning curves were generated via the following:
```
python -m dialogue_kt.main visualize --dataset eedi --tag_src <type of KC used> --model_name <trained model to visualize predictions for>
```
The learning curves were saved in the `results` folder as well.