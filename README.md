PaliGemma with QLoRA
This repository demonstrates fine-tuning and optimization of the PaliGemma model using the QLoRA (Quantized Low-Rank Adaptation) technique. The focus is on enabling efficient training of large language models through low-rank adaptations and quantization methods.

Installation
To get started, clone the Hugging Face Transformers repository and install it:

bash
Copy code
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
Ensure that Python version 3.9.19 is installed.

Notebook Overview
1. Imports and Initialization
Key Libraries:
The notebook uses torch, datasets, transformers, and peft for fine-tuning and optimization tasks.
Model:
The base model used is google/paligemma-3b-pt-224.
Device Selection:
Dynamically selects cuda for GPU or cpu for training.
2. Dataset Handling
The dataset is loaded for training and evaluation using datasets.load_dataset.
Screenshot


Key Features
Utilizes QLoRA to enable efficient training of large language models.
Demonstrates dynamic device selection for optimized performance.
Provides a streamlined process for fine-tuning the PaliGemma model.
Requirements
Python 3.9.19
PyTorch
Hugging Face Transformers Library
Datasets Library
Usage
To run the notebook:

Clone this repository.
Install the necessary dependencies as described above.
Open the notebook in Jupyter or any compatible environment and execute the cells.

