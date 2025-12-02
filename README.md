# GPT-2 Fine-tuning on Custom Dataset (Ye + Trump Quotes)

**Note:** This code is adapted from another source with modifications to the dataset.
### [Google Colab](https://colab.research.google.com/drive/1m2mQy20kvW3UdKNK488HwRuGjctXtgri?usp=sharing)

## Architecture Overview

This project fine-tunes OpenAI's GPT-2 language model using the Hugging Face Transformers library to generate text in the style of Kanye West and Donald Trump quotes. The implementation uses Parameter-Efficient Fine-Tuning (PEFT) techniques and includes post-training optimizations for deployment.

## System Components

### Data Processing Pipeline
The notebook merges two text files (`kanye-quotes.txt` and `trump-quotes.txt`) into a single training corpus. The combined dataset is tokenized using GPT-2's BPE tokenizer with a maximum sequence length of 128 tokens. Text sequences are truncated and padded to ensure uniform input dimensions during training.

### Model Configuration
The base model is GPT-2 (124M parameters) with the following specifications:
- 12 transformer layers with 768-dimensional embeddings
- 12 attention heads per layer
- 3072-dimensional feed-forward networks
- Vocabulary size of 50,257 tokens
- Context window of 1,024 tokens (training) expandable to 4,096 (inference)

### Training Setup
The fine-tuning process uses causal language modeling with cross-entropy loss. Training hyperparameters include 2 epochs, batch size of 4 with 4 gradient accumulation steps (effective batch size of 16), learning rate of 3e-4, and FP16 mixed precision training for efficiency. The `DataCollatorForLanguageModeling` handles dynamic batching without masked language modeling.

### Model Optimization & Export
After training, the notebook applies three optimization techniques:

**Weight Sparsification**: 50% of linear layer weights are randomly zeroed to reduce model size while maintaining reasonable performance.

**Quantization**: The model is converted to GGUF format using llama.cpp's conversion tools and quantized to Q2_K format (2-bit weights with k-quants) for efficient inference. This reduces the model from ~500MB to approximately 78MB.

**4-bit Loading**: BitsAndBytes configuration enables loading the model in 4-bit precision using NF4 quantization for memory-efficient deployment.

## Technical Workflow

The complete pipeline executes in six stages:
1. Install dependencies (transformers, datasets, torch, PEFT, bitsandbytes)
2. Merge and prepare training data
3. Load and tokenize dataset with GPT-2 tokenizer
4. Fine-tune model using Hugging Face Trainer API
5. Apply sparsification to model weights
6. Convert to GGUF format and quantize for deployment

## Output
The final model is saved as `gpt2-ye-trump-Q2_K.gguf`, a highly compressed quantized model suitable for CPU inference on resource-constrained devices.