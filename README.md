# 🌟 Image Captioning with Transformer Decoder + ResNet-50 Encoder

This project implements an end-to-end **Image Caption Generator** using a **Transformer Decoder** and a **pre-trained ResNet-50 CNN Encoder**. It includes support for **beam search decoding**, **CUDA acceleration**, and is deployed using **Gradio** for real-time caption generation.

---

## 💡 What This Project Does

Given an input image, the model generates a human-like caption describing its content. It combines:

- 🧠 **Visual understanding** via **ResNet-50**
- ✍️ **Sequence generation** via **Transformer Decoder**
- 🛤️ **Caption enhancement** using **Beam Search decoding**

---

## 🎨 Sample Outputs

| Image | Generated Caption |
|-------|-------------------|
| 🐶 Dog | "a large brown dog laying on top of a grass covered field" |

---
![Screenshot%202025-06-23](Screenshot%20091543.png)

## 📊 BLEU Score Performance (Top 100 Validation Samples)

| Metric   | Score  |
|----------|--------|
| BLEU-1   | 0.3541 |
| BLEU-2   | 0.2027 |
| BLEU-3   | 0.1269 |
| BLEU-4   | 0.0853 |

> These scores reflect competitive accuracy for a lightweight model trained on limited compute.

---

## 🧰 Architecture Overview

### 📘 Encoder (ResNet-50 CNN)

```python
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
features = nn.Sequential(*list(resnet.children())[:-2])
features = nn.Conv2d(2048, embed_size, kernel_size=1)(features)
```
Uses pretrained ResNet-50 for image feature extraction

Final fully-connected layer removed

Feature map reshaped to [sequence_len, batch_size, embed_dim]

### 🌐 Decoder (Transformer)

```python
embedding = nn.Embedding(vocab_size, embed_size)
positional_encoding = PositionalEncoding(embed_size)
decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads=4, dim_feedforward=512)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
```
Multi-head self-attention captures temporal dependencies

Positional Encoding maintains word order

Final linear layer maps to vocab logits

### 🧠 Beam Search Decoding

```python
log_probs = torch.nn.functional.log_softmax(output[-1, 0], dim=0)
topk = torch.topk(log_probs, beam_size)
```
Keeps top-k sequences during decoding

Final caption selected based on highest cumulative log probability

🚀 Training Pipeline
🗂 Dataset
MS COCO 2017 Captions

Each image has 5 human-annotated captions

Loaded using torchvision.datasets.CocoCaptions

🧮 Optimization
```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
```

CrossEntropyLoss with padding mask

Adam Optimizer with gradient clipping

AMP training using torch.cuda.amp for mixed precision

🏋️‍♀️ Strategy
50 epochs with early stopping and checkpointing

Vocabulary built using word frequency threshold

Trained on NVIDIA GPU (CUDA)

### 🖼️ Deployment via Gradio
```python
def generate_caption(image):
    tensor = transform(image).to(device)
    caption = generate_caption_beam_search(decoder, encoder, tensor, vocab)
    return caption
```

UI built with gr.Interface

Drag-and-drop image upload

Returns caption in real-time


📲 Real-World Applications
🧏 Accessibility: Describe visual elements for visually impaired/deaf users via speech/text

🤖 Robotics: Let robots explain surroundings using language

📸 Surveillance: Caption camera footage for automated logging

🛒 E-commerce: Improve tagging and image-based search relevance

⚙️ Tech Stack Breakdown

| Component       | Tool / Library            | Code Integration / Use Case                                      |
|----------------|----------------------------|------------------------------------------------------------------|
| Dataset         | MS COCO                    | `torchvision.datasets.CocoCaptions()`                            |
| Deep Learning   | PyTorch                    | Core model logic & training loop                                 |
| CNN Encoder     | ResNet-50 (torchvision)    | `models.resnet50(weights=...)` + `Conv2D` to embedding           |
| Decoder         | TransformerDecoder         | `nn.TransformerDecoder`, 2-layer, multi-head                     |
| Preprocessing   | Custom Vocabulary          | Tokenize, numericalize, decode captions                          |
| Optimization    | Adam + AMP                 | `torch.optim.Adam` + `torch.cuda.amp.autocast` for acceleration  |
| Caption Logic   | Beam Search                | Keeps top-k most probable sequences                              |
| Interface       | Gradio                     | `gr.Interface(fn=..., inputs=..., outputs=...)`                  |
| Hardware Accel. | CUDA + AMP                 | Enabled using `torch.device("cuda")`                             |


📫 Contact
If you liked this project or want to collaborate:

Sumit Dighe
Email - Sdighe.personal@gmail.com
