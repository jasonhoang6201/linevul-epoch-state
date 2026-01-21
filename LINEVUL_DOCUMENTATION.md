# LineVul - Technical Documentation

## Tổng quan toàn diện về Transformer-based Line-Level Vulnerability Prediction

---

## Mục lục

1. [Giới thiệu và Idea chính](#1-giới-thiệu-và-idea-chính)
2. [Research Motivation - Tại sao cần LineVul?](#2-research-motivation---tại-sao-cần-linevul)
3. [Kiến trúc tổng thể (Architecture)](#3-kiến-trúc-tổng-thể-architecture)
4. [Core Model - Tại sao chọn RoBERTa/CodeBERT?](#4-core-model---tại-sao-chọn-robertacodebert)
5. [Training Flow](#5-training-flow)
6. [Inference Flow](#6-inference-flow)
7. [Line-Level Localization - Explainability Methods](#7-line-level-localization---explainability-methods)
8. [Thư viện và Dependencies](#8-thư-viện-và-dependencies)
9. [Dataset và Data Processing](#9-dataset-và-data-processing)
10. [Tokenization Strategies](#10-tokenization-strategies)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Ablation Study Insights](#12-ablation-study-insights)
13. [So sánh với Baseline Methods](#13-so-sánh-với-baseline-methods)
14. [Cấu trúc thư mục Source Code](#14-cấu-trúc-thư-mục-source-code)
15. [Key Takeaways và Lưu ý quan trọng](#15-key-takeaways-và-lưu-ý-quan-trọng)

---

## 1. Giới thiệu và Idea chính

### 1.1 Problem Statement

**Vulnerability Detection** trong source code là một bài toán cực kỳ quan trọng trong Software Security. Trước LineVul, các phương pháp truyền thống chỉ tập trung vào:
- **Function-level prediction**: Chỉ xác định hàm có chứa lỗi hay không
- **Không cung cấp line-level localization**: Developer phải tự tìm dòng code chứa lỗi

### 1.2 Core Idea của LineVul

LineVul giới thiệu một cách tiếp cận mới:

```
Input: Source code function (C/C++)
       |
       v
   [BPE Tokenization]
       |
       v
   [Pre-trained CodeBERT/RoBERTa]
       |
       v
   [Classification Head]
       |
       v
Output 1: Function-level prediction (vulnerable/non-vulnerable)
Output 2: Line-level vulnerability localization (using attention scores)
```

**Key Innovation**: Sử dụng **Self-Attention scores** từ Transformer để xác định dòng code nào có khả năng chứa vulnerability cao nhất.

### 1.3 Tại sao Idea này hiệu quả?

1. **Transfer Learning**: Sử dụng pre-trained model trên code (CodeBERT) giúp hiểu semantic của code
2. **Attention as Explanation**: Attention mechanism cung cấp interpretability tự nhiên
3. **End-to-end Learning**: Không cần feature engineering thủ công

---

## 2. Research Motivation - Tại sao cần LineVul?

### 2.1 Limitations của các phương pháp trước

| Phương pháp | Vấn đề |
|-------------|--------|
| Static Analysis (CppCheck) | False positive cao, chỉ detect pattern đã biết |
| Deep Learning trước (VulDeePecker, SySeVR) | F1-score thấp (~0.19-0.27) |
| Graph-based (Devign, IVDetect) | Phức tạp, cần xây dựng graph |
| BoW + Random Forest | Không hiểu semantic, F1 chỉ 0.25 |

### 2.2 LineVul Advantages

1. **Superior Performance**: F1 = 0.91 (so với 0.35 của IVDetect)
2. **Line-level Localization**: Top-10 Accuracy = 65%
3. **Interpretable**: Attention scores có thể hiểu được
4. **Efficient**: Không cần xây dựng complex graph representations

---

## 3. Kiến trúc tổng thể (Architecture)

### 3.1 High-level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LineVul Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ Source Code  │───>│  BPE Tokenizer   │───>│  Input IDs     │ │
│  │ (Function)   │    │ (CodeBERT's)     │    │ [512 tokens]   │ │
│  └──────────────┘    └──────────────────┘    └───────┬────────┘ │
│                                                       │          │
│                                                       v          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              RoBERTa Encoder (12 layers, 12 heads)          │ │
│  │                                                              │ │
│  │  - Pre-trained on CodeSearchNet (6 programming languages)   │ │
│  │  - Hidden size: 768                                          │ │
│  │  - Vocab size: 50265                                         │ │
│  └───────────────────────────┬────────────────────────────────┘ │
│                               │                                  │
│              ┌────────────────┴────────────────┐                │
│              v                                  v                │
│  ┌──────────────────────┐          ┌──────────────────────────┐ │
│  │  Classification Head │          │  Attention Weights       │ │
│  │  - Dense(768, 768)   │          │  (for line localization) │ │
│  │  - Tanh activation   │          │                          │ │
│  │  - Dropout           │          │                          │ │
│  │  - Dense(768, 2)     │          │                          │ │
│  └──────────┬───────────┘          └────────────┬─────────────┘ │
│             │                                    │               │
│             v                                    v               │
│  ┌──────────────────────┐          ┌──────────────────────────┐ │
│  │ Softmax Probability  │          │  Line-level Scores       │ │
│  │ [P(vul), P(non-vul)] │          │  (sum attention per line)│ │
│  └──────────────────────┘          └──────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Class Definition

Từ file `linevul_model.py`:

```python
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 768 -> 768
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)  # 768 -> 2 (binary classification)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(RobertaForSequenceClassification):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
```

**Key Points**:
- Sử dụng **[CLS] token** (position 0) để represent toàn bộ function
- **Binary classification**: Vulnerable (1) vs Non-vulnerable (0)
- Dropout để tránh overfitting

---

## 4. Core Model - Tại sao chọn RoBERTa/CodeBERT?

### 4.1 CodeBERT là gì?

**CodeBERT** là một pre-trained model được train trên:
- **CodeSearchNet corpus**: 6 programming languages (Python, Java, JavaScript, Go, Ruby, PHP)
- **2.3 million functions** paired with natural language documentation
- **Masked Language Modeling (MLM)** + **Replaced Token Detection (RTD)**

### 4.2 Tại sao CodeBERT phù hợp cho Vulnerability Detection?

| Đặc điểm | Lợi ích |
|----------|---------|
| Pre-trained on code | Đã học được syntax và semantics của programming languages |
| RoBERTa architecture | State-of-the-art Transformer architecture |
| BPE tokenization | Xử lý tốt out-of-vocabulary tokens trong code |
| Large vocabulary | 50,265 tokens, bao phủ nhiều code patterns |

### 4.3 Tại sao không dùng BERT gốc?

1. **BERT** được train trên natural language (Wikipedia, books)
2. **CodeBERT** được train trên CODE - hiểu được:
   - Variable naming conventions
   - Function signatures
   - Code structure (loops, conditions, etc.)
   - API usage patterns

### 4.4 Ablation Study Results chứng minh

```
| Model Configuration                        |  F1  |
|--------------------------------------------|------|
| BPE + Pre-training (CodeBERT) + BERT       | 0.91 | <-- Best (LineVul)
| BPE + No Pre-training + BERT               | 0.80 |
| Word-level + Pre-training (CodeBERT) + BERT| 0.42 |
| Word-level + No Pre-training + BERT        | 0.39 |
```

**Conclusion**: Pre-training và BPE tokenization đều cực kỳ quan trọng!

---

## 5. Training Flow

### 5.1 Training Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Data Loading                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ - Load train.csv (150,908 samples)                   │    │
│  │ - Load val.csv (18,864 samples)                      │    │
│  │ - Columns: processed_func, target, vul_func_with_fix │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          v                                   │
│  Step 2: Tokenization                                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ - Tokenize each function using CodeBERT tokenizer    │    │
│  │ - Truncate to 512 tokens (block_size)                │    │
│  │ - Add special tokens: <s> ... </s>                   │    │
│  │ - Pad to fixed length                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          v                                   │
│  Step 3: Training Loop                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ For each epoch (10 epochs):                          │    │
│  │   For each batch (batch_size=16):                    │    │
│  │     1. Forward pass: model(input_ids, labels)        │    │
│  │     2. Compute CrossEntropyLoss                      │    │
│  │     3. Backward pass: loss.backward()                │    │
│  │     4. Gradient clipping: max_grad_norm=1.0          │    │
│  │     5. Optimizer step: AdamW (lr=2e-5)               │    │
│  │     6. Scheduler step: linear warmup                 │    │
│  │   End batch                                          │    │
│  │   Evaluate on validation set                         │    │
│  │   Save if best F1                                    │    │
│  │ End epoch                                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

### 5.2 Training Hyperparameters

```python
# Từ linevul_main.py và train.log
epochs = 10
train_batch_size = 16
eval_batch_size = 16
learning_rate = 2e-5
max_grad_norm = 1.0
block_size = 512  # max sequence length
warmup_steps = max_steps // 5
weight_decay = 0.0
adam_epsilon = 1e-8
seed = 123456
```

### 5.3 Optimizer và Scheduler

```python
# AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

# Linear warmup scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_steps
)
```

### 5.4 Training Progress (từ train.log)

```
Epoch 0: loss=0.113, eval_f1=0.877
Epoch 1: loss=0.063, eval_f1=0.887
Epoch 2: loss=0.054, eval_f1=0.906
Epoch 3: loss=0.045, eval_f1=0.915 <-- Best checkpoint saved
Epoch 4: loss=0.035, eval_f1=0.912
Epoch 5: loss=0.027, eval_f1=0.917
...
```

---

## 6. Inference Flow

### 6.1 Function-level Prediction

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: New source code function                              │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 1. Tokenize using CodeBERT BPE tokenizer                 │ │
│  │    code_tokens = tokenizer.tokenize(func)[:510]          │ │
│  │    source_tokens = [<s>] + code_tokens + [</s>]          │ │
│  │    source_ids = tokenizer.convert_tokens_to_ids(...)     │ │
│  │    Pad to 512 tokens                                     │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 2. Forward pass through model                            │ │
│  │    with torch.no_grad():                                 │ │
│  │        loss, logits = model(input_ids, labels)           │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 3. Apply softmax and threshold                           │ │
│  │    prob = softmax(logits)                                │ │
│  │    y_pred = prob[:, 1] > 0.5  # threshold                │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  Output: 0 (non-vulnerable) hoặc 1 (vulnerable)               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Line-level Localization (khi function được predict là vulnerable)

```
┌─────────────────────────────────────────────────────────────┐
│               LINE-LEVEL LOCALIZATION                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Nếu function được predict là VULNERABLE:                     │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 1. Extract attention weights                             │ │
│  │    prob, attentions = model(input_ids, output_attentions)│ │
│  │    attentions: shape (12 layers, 12 heads, 512, 512)     │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 2. Aggregate attention scores                            │ │
│  │    For each layer:                                       │ │
│  │        Sum attention values across heads                 │ │
│  │    Sum across all layers                                 │ │
│  │    Result: 1D tensor [512] - score per token             │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 3. Map tokens back to lines                              │ │
│  │    Group tokens by newline character (Ċ)                 │ │
│  │    Sum attention scores for each line                    │ │
│  │    Result: List of line scores                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 4. Rank lines by score                                   │ │
│  │    Sort lines descending by attention score              │ │
│  │    Top-K lines = most likely vulnerable                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│         │                                                     │
│         v                                                     │
│  Output: Ranked list of lines by vulnerability likelihood    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Line-Level Localization - Explainability Methods

### 7.1 Các phương pháp được implement

LineVul implement 6 phương pháp explainability khác nhau:

| Method | Library | Description |
|--------|---------|-------------|
| **Self-Attention** | Built-in | Trực tiếp sử dụng attention weights từ Transformer |
| **Layer Integrated Gradient (LIG)** | Captum | Tính gradient của output theo input, tích hợp qua các layer |
| **Saliency** | Captum | Gradient-based attribution |
| **DeepLift** | Captum | So sánh activation với reference input |
| **DeepLiftShap** | Captum | Kết hợp DeepLift với Shapley values |
| **GradientShap** | Captum | Gradient-based Shapley values |

### 7.2 Self-Attention Method (Best performing)

```python
# Từ linevul_main.py:line_level_localization_tp()
def get_attention_scores(model, input_ids):
    prob, attentions = model(input_ids, output_attentions=True)
    # attentions: tuple of (batch, num_heads, seq_len, seq_len)
    attentions = attentions[0][0]  # Get first layer, first batch
    
    # Sum across all layers and heads
    attention = None
    for layer_attention in attentions:
        layer_attention = sum(layer_attention)  # Sum across heads
        if attention is None:
            attention = layer_attention
        else:
            attention += layer_attention
    
    return attention  # Shape: [512]
```

### 7.3 Performance Comparison

```
| Method                    | Top-10 Accuracy | IFA (Initial False Alarm) |
|---------------------------|-----------------|---------------------------|
| Self-Attention            | 0.65            | 4.56                      |
| Layer Integrated Gradient | 0.53            | 8.31                      |
| Saliency                  | 0.58            | 6.93                      |
| DeepLift                  | 0.57            | 6.27                      |
| DeepLiftShap              | 0.57            | 6.26                      |
| GradientShap              | 0.52            | 7.82                      |
| CppCheck (baseline)       | 0.15            | 21.6                      |
```

**Key Insight**: Self-Attention cho kết quả tốt nhất vì nó trực tiếp phản ánh những gì model "chú ý" khi đưa ra prediction.

---

## 8. Thư viện và Dependencies

### 8.1 Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **torch** | 1.13.1 | Deep learning framework |
| **transformers** | 4.26.0 | Pre-trained models (RoBERTa, CodeBERT) |
| **tokenizers** | 0.13.2 | Fast tokenization |
| **captum** | 0.7.0 | Model interpretability/explainability |
| **scikit-learn** | 1.2.1 | Metrics, baseline models |
| **pandas** | 1.5.2 | Data processing |
| **numpy** | 1.24.2 | Numerical computing |

### 8.2 Tại sao cần các thư viện này?

#### 8.2.1 PyTorch + Transformers
```python
from transformers import (
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
```
- **RobertaConfig**: Cấu hình model architecture
- **RobertaForSequenceClassification**: Pre-trained model
- **RobertaTokenizer**: BPE tokenizer
- **get_linear_schedule_with_warmup**: Learning rate scheduling

#### 8.2.2 Captum
```python
from captum.attr import (
    LayerIntegratedGradients,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    Saliency
)
```
- Cung cấp các attribution methods để explain model predictions
- Thiết yếu cho line-level localization

#### 8.2.3 Scikit-learn
```python
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
```
- Evaluation metrics
- Baseline model (BoW + Random Forest)

---

## 9. Dataset và Data Processing

### 9.1 Big-Vul Dataset

| Property | Value |
|----------|-------|
| Source | Real-world C/C++ vulnerabilities |
| Train | 150,908 samples |
| Validation | 18,864 samples |
| Test | 27,818 samples |
| Total | ~197,590 samples |

### 9.2 Data Columns

```
| Column | Type | Description |
|--------|------|-------------|
| processed_func | str | Original C/C++ function source code |
| target | int | 0 = non-vulnerable, 1 = vulnerable |
| vul_func_with_fix | str | Fixed function with marked changes |
| flaw_line | str | Actual vulnerable lines (separated by /~/) |
| flaw_line_index | str | Indices of vulnerable lines |
```

### 9.3 Data Processing Flow

```python
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        df = pd.read_csv(file_path)
        funcs = df["processed_func"].tolist()
        labels = df["target"].tolist()
        
        for i in range(len(funcs)):
            self.examples.append(
                convert_examples_to_features(funcs[i], labels[i], tokenizer, args)
            )
```

### 9.4 Feature Conversion

```python
def convert_examples_to_features(func, label, tokenizer, args):
    # Tokenize
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size-2]  # Leave room for special tokens
    
    # Add special tokens
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    
    # Convert to IDs
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    
    # Pad to fixed length
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    
    return InputFeatures(source_tokens, source_ids, label)
```

---

## 10. Tokenization Strategies

### 10.1 Ba loại Tokenizer được experiment

#### 10.1.1 BPE Tokenizer (Pre-trained - CodeBERT)
```python
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
```
- **Sử dụng trong LineVul chính**
- Vocabulary: 50,265 tokens
- Đã được train trên code corpus

#### 10.1.2 BPE Tokenizer (Custom trained)
```python
# train_bpe_tokenizer.py
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files="../data/tokenizer_train_data.txt",
    vocab_size=50257,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
```

#### 10.1.3 Word-level Tokenizer
```python
# train_word_level_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
trainer = WordLevelTrainer(
    vocab_size=50257,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2
)
```

### 10.2 Tại sao BPE tốt hơn Word-level cho code?

| Aspect | BPE | Word-level |
|--------|-----|------------|
| OOV handling | Phân tách thành subwords | Thay bằng [UNK] |
| Variable names | `myVariableName` -> `my`, `Variable`, `Name` | `[UNK]` |
| Vocabulary efficiency | Cao | Thấp |
| Performance (F1) | 0.91 | 0.42 |

**Ví dụ**:
```
Input: "getUserAccountById"

BPE: ["get", "User", "Account", "By", "Id"]
Word-level: ["getUserAccountById"] hoặc ["[UNK]"]
```

---

## 11. Evaluation Metrics

### 11.1 Function-level Metrics

| Metric | Formula | LineVul Result |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | 0.97 |
| **Recall** | TP / (TP + FN) | 0.86 |
| **F1-Score** | 2 * P * R / (P + R) | 0.91 |
| **Accuracy** | (TP + TN) / Total | - |

### 11.2 Line-level Metrics

#### 11.2.1 Top-K Accuracy
```
Top-K Accuracy = (# functions với ít nhất 1 flaw line trong top-K) / (tổng functions)
```

| K | LineVul | CppCheck |
|---|---------|----------|
| 1 | 10% | 7% |
| 3 | 31% | 9% |
| 5 | 46% | 12% |
| 10 | 65% | 15% |

#### 11.2.2 Initial False Alarm (IFA)
```
IFA = Số dòng clean trung bình phải kiểm tra trước khi tìm được flaw line đầu tiên
```
- LineVul (Attention): **4.56 dòng**
- CppCheck: **21.6 dòng**

#### 11.2.3 Effort@TopK%Recall
```
Effort = (# dòng phải kiểm tra để tìm K% flaw lines) / (tổng số dòng)
```

#### 11.2.4 Recall@TopK%LOC
```
Recall = (# flaw lines trong top K% dòng) / (tổng flaw lines)
```

---

## 12. Ablation Study Insights

### 12.1 Key Findings

```
┌────────────────────────────────────────────────────────────────┐
│                    ABLATION STUDY RESULTS                       │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Đóng góp của mỗi component vào F1 score:                        │
│                                                                  │
│  Full LineVul (BPE + PreTrain + BERT)         ████████████ 0.91 │
│                                                                  │
│  Bỏ Pre-training                                                 │
│  (BPE + NoPretrain + BERT)                    █████████░░░ 0.80 │
│  --> Pre-training đóng góp +0.11 F1                              │
│                                                                  │
│  Dùng Word-level thay vì BPE                                     │
│  (WordLevel + PreTrain + BERT)                █████░░░░░░░ 0.42 │
│  --> BPE đóng góp +0.49 F1                                       │
│                                                                  │
│  Bỏ cả hai                                                       │
│  (WordLevel + NoPretrain + BERT)              ████░░░░░░░░ 0.39 │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

### 12.2 Interpretation

1. **BPE Tokenization quan trọng nhất** (+0.49 F1)
   - Xử lý tốt variable names, function names
   - Giảm OOV rate

2. **Pre-training trên code cũng quan trọng** (+0.11 F1)
   - Transfer learning từ CodeSearchNet
   - Model đã hiểu semantic của code

3. **Cả hai kết hợp tạo synergy**
   - 0.39 -> 0.91 (cải thiện 0.52 F1)

---

## 13. So sánh với Baseline Methods

### 13.1 BoW + Random Forest Implementation

Từ `bow_rf/rf_main.py`:

```python
# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(norm='l2', max_features=1000)
X_train = vectorizer.fit_transform(train_data["processed_func"])

# Random Forest classifier
rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rf.fit(X_train, y_train)
```

**Result**: F1 = 0.25 (so với LineVul 0.91)

### 13.2 CppCheck (Static Analysis)

Từ `cppcheck/run.py`:

```python
# Parse CppCheck output
stc_result = static_results[i].split("\n")
stc_result = [x for x in stc_result if "error:" in x]

# Match with ground truth
for err in errs:
    if err in flaw_lines_truth:
        correct_count += 1
```

**Results**:
- Top-10 Accuracy: 15% (vs LineVul 65%)
- IFA: 21.6 (vs LineVul 4.56)

### 13.3 Comparison Table

```
┌───────────────────┬──────┬───────────┬────────┐
│      Model        │  F1  │ Precision │ Recall │
├───────────────────┼──────┼───────────┼────────┤
│ LineVul           │ 0.91 │   0.97    │  0.86  │
│ IVDetect          │ 0.35 │   0.23    │  0.72  │
│ Reveal            │ 0.30 │   0.19    │  0.74  │
│ SySeVR            │ 0.27 │   0.15    │  0.74  │
│ Devign            │ 0.26 │   0.18    │  0.52  │
│ BoW+RF            │ 0.25 │   0.48    │  0.17  │
│ Russell et al.    │ 0.24 │   0.16    │  0.48  │
│ VulDeePecker      │ 0.19 │   0.12    │  0.49  │
└───────────────────┴──────┴───────────┴────────┘
```

---

## 14. Cấu trúc thư mục Source Code

```
LineVul/
├── README.md                    # Documentation chính
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
│
├── linevul/                     # Main implementation
│   ├── linevul_main.py         # Training/inference pipeline (1254 dòng)
│   ├── linevul_model.py        # Model architecture (60 dòng)
│   ├── train_bpe_tokenizer.py  # Custom BPE tokenizer training
│   ├── train_word_level_tokenizer.py  # Word-level tokenizer training
│   │
│   ├── bpe_tokenizer/          # Trained BPE tokenizer files
│   │   ├── bpe_tokenizer-vocab.json
│   │   ├── bpe_tokenizer-merges.txt
│   │   └── config.json
│   │
│   ├── word_level_tokenizer/   # Trained word-level tokenizer
│   │   └── wordlevel.json
│   │
│   ├── saved_models/           # Model checkpoints
│   │   └── checkpoint-best-f1/
│   │       └── 12heads_linevul_model.bin
│   │
│   ├── results/                # Evaluation results
│   │
│   ├── ifa_records/            # IFA evaluation records
│   │   ├── ifa_attention.txt
│   │   ├── ifa_lig.txt
│   │   ├── ifa_deeplift.txt
│   │   └── ...
│   │
│   └── train_logs/             # Training logs
│       ├── linvul_train.log
│       └── ...
│
├── bow_rf/                      # Baseline: Bag-of-Words + Random Forest
│   └── rf_main.py
│
├── cppcheck/                    # Baseline: Static Analysis comparison
│   ├── run.py                  # Main evaluation script
│   ├── write_static_analysis_data.py
│   ├── output_to_results.py
│   └── data/
│       ├── static_analysis_data.csv
│       ├── static_analysis_results.csv
│       └── c_files/            # Individual C++ files for CppCheck
│
├── data/                        # Dataset
│   └── big-vul_dataset/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
└── logo/                        # Project logos
    ├── linevul_logo.png
    └── msr_cover.png
```

---

## 15. Key Takeaways và Lưu ý quan trọng

### 15.1 Điểm mạnh của LineVul

1. **State-of-the-art Performance**: F1 = 0.91, vượt trội hơn các phương pháp trước
2. **Line-level Localization**: Xác định được dòng code chứa lỗi với độ chính xác 65%
3. **Interpretable**: Attention scores có thể giải thích được
4. **Efficient**: Không cần xây dựng complex representations (graph, AST)

### 15.2 Hạn chế

1. **Sequence Length Limit**: Chỉ xử lý được 512 tokens (~100-200 dòng code)
2. **C/C++ Only**: Train trên Big-Vul dataset, chủ yếu là C/C++
3. **GPU Required**: BERT-based model cần GPU để train (8GB+ VRAM)

### 15.3 Khi nào nên sử dụng LineVul?

- **Nên dùng khi**:
  - Cần detect vulnerabilities trong C/C++ code
  - Cần biết chính xác dòng nào chứa lỗi
  - Có đủ GPU resources

- **Không nên dùng khi**:
  - Functions quá dài (>512 tokens)
  - Ngôn ngữ khác C/C++ (cần re-train)
  - Resource-constrained environments

### 15.4 Best Practices khi sử dụng

```bash
# Inference với pre-trained model
python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --do_local_explanation \
  --reasoning_method=attention \
  --test_data_file=your_test_data.csv \
  --block_size 512 \
  --eval_batch_size 512
```

### 15.5 Tips để cải thiện hiệu suất

1. **Data**: Thêm training data cho domain của bạn
2. **Fine-tuning**: Fine-tune trên specific vulnerability types
3. **Ensemble**: Kết hợp nhiều explanation methods
4. **Post-processing**: Lọc kết quả dựa trên code structure

---

## Appendix A: Command Reference

### A.1 Training

```bash
python linevul_main.py \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=../data/big-vul_dataset/train.csv \
  --eval_data_file=../data/big-vul_dataset/val.csv \
  --test_data_file=../data/big-vul_dataset/test.csv \
  --epochs 10 \
  --block_size 512 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456
```

### A.2 Inference Only

```bash
python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --test_data_file=../data/big-vul_dataset/test.csv \
  --block_size 512 \
  --eval_batch_size 512
```

### A.3 Line-level Explanation

```bash
python linevul_main.py \
  --model_name=12heads_linevul_model.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --do_local_explanation \
  --top_k_constant=10 \
  --reasoning_method=all \
  --test_data_file=../data/big-vul_dataset/test.csv \
  --block_size 512 \
  --eval_batch_size 512
```

---

## Appendix B: Citation

```bibtex
@inproceedings{fu2022linevul,
  title={LineVul: A Transformer-based Line-Level Vulnerability Prediction},
  author={Fu, Michael and Tantithamthavorn, Chakkrit},
  booktitle={2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR)},
  year={2022},
  organization={IEEE}
}
```

---

*Document được tạo từ phân tích source code của LineVul repository.*
*Cập nhật lần cuối: Tháng 1, 2026*
