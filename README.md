
# Emotion Text Classification Using Llama3-8b and LoRA

## Introduction

This project explores emotion text classification using the Llama3-8b model, enhanced with LoRA and FlashAttention techniques. The model is optimized for identifying six emotion categories: joy, sadness, anger, fear, love, and surprise. The Llama3-8b model demonstrates superior performance with an accuracy of 0.9262, surpassing other transformer models such as Bert-Base, Bert-Large, Roberta-Base, and Roberta-Large.

## Background

Natural Language Processing (NLP) has become a key focus area for sentiment analysis, also known as sentiment classification or sentiment detection. This technology helps businesses understand consumer emotions and opinions, enhancing customer satisfaction and product development. The vast amount of data in large companies makes manual analysis impractical, leading to the adoption of AI and NLP algorithms.

## Key Features

- **Model**: Llama3-8b, fine-tuned using supervised learning.
- **Techniques**: Utilized LoRA for efficient parameter tuning and FlashAttention for optimized attention computation.
- **Dataset**: Emotion text dataset with six categories.
- **Performance**: Achieved an accuracy of 0.9262, surpassing other NLP models.

## Methods

### Llama3-8b Model

Llama3-8b is a large language model developed by Meta AI, featuring 8 billion parameters. It is designed for dialogue use cases and includes advancements such as Grouped-Query Attention (GQA), which optimizes memory and computational efficiency.

**Table 1: Llama3-8b Model Details**
| Feature                | Specification          |
|------------------------|------------------------|
| Training Data          | Publicly available data|
| Parameters             | 8B                     |
| Context Length         | 8k                     |
| GQA                    | Yes                    |
| Token Count            | 15T+                   |
| Knowledge Cutoff       | March 2023             |

**Figure 1: Architecture of Llama3-8b**
![fig1](fig1.png)

### LoRA Technique

LoRA integrates trainable low-rank matrices into each Transformer layer, significantly reducing the number of trainable parameters while keeping the main model weights unchanged. This approach enhances training efficiency and reduces storage needs without increasing inference latency.

**Figure 2: LoRA Training Method**
![fig2](fig2.png)

### FlashAttention V2

FlashAttention optimizes the attention mechanism in Transformer models by enhancing computational efficiency and reducing memory usage. It uses block-wise computation and sparse matrix operations to improve cache utilization and minimize processing time.

## Experimentation

### Data Analysis

The dataset includes six emotions: joy, sadness, anger, fear, love, and surprise. The distribution is relatively balanced, with "Joy" being the most common and "Surprise" the least common emotion.

**Figure 3: Emotion Text Label Distribution**
![fig3](fig3.png)

### Experiment Settings

- **Optimizer**: Adam
- **Learning Rate**: 5e-5
- **Batch Size**: 5
- **Epochs**: 3
- **LoRA Rank**: 8
- **Gradient Accumulation Steps**: 4
- **Max Length**: 512 tokens
- **Precision**: FP16 for reduced GPU memory usage

**Table 2: Experiment Settings for Llama3-8b**
| Parameter                    | Setting               |
|------------------------------|-----------------------|
| Optimizer                    | Adam                  |
| Learning Rate                | 5e-5                  |
| Batch Size                   | 5                     |
| Epochs                       | 3                     |
| LoRA Rank                    | 8                     |
| Gradient Accumulation Steps  | 4                     |
| Max Length                   | 512                   |

The Adam optimizer was used for its adaptive learning rate capabilities, combined with a cosine learning rate schedule. FP16 precision was employed to save GPU memory.

### Evaluation Metrics

The primary metric used for evaluation is accuracy, defined as:

\[
	ext{Accuracy} = rac{	ext{TP} + 	ext{FN}}{	ext{TP} + 	ext{FP} + 	ext{FN} + 	ext{TN}}
\]

Where:
- TP = True Positive
- FP = False Positive
- FN = False Negative
- TN = True Negative

**Table 3: Accuracy Results for Different Models**
| Model           | Accuracy |
|-----------------|----------|
| Bert-Base       | 0.9063   |
| Bert-Large      | 0.9086   |
| Roberta-Base    | 0.9125   |
| Roberta-Large   | 0.9189   |
| Llama3-8b       | 0.9262   |

## Conclusion

This project demonstrates the potential of large language models, such as Llama3-8b, in domain-specific tasks like emotion text classification. The model's performance, boosted by specialized techniques like LoRA and FlashAttention, underscores the effectiveness of large models in achieving high accuracy in NLP applications.

## Contact

For any questions or issues, please feel free to contact us.
