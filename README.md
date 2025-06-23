"""# 🌟 Image Captioning with Transformer Decoder + ResNet-50 Encoder

This project implements an end-to-end **Image Caption Generator** using a **Transformer Decoder** and a pre-trained **ResNet-50 CNN Encoder**. It supports **Beam Search decoding**, is trained on the **MS COCO dataset**, uses **CUDA acceleration**, and is deployed as a **Gradio app** for real-time caption generation from images.

---

## 💡 What This Project Does

Given an input image, the model generates a natural language caption. It combines visual understanding via ResNet-50 feature extractor, sequence generation via Transformer Decoder, and caption enhancement using Beam Search decoding.

---

## 🎨 Sample Outputs

| Image      | Generated Caption                                                  |
|------------|--------------------------------------------------------------------|
| 🐶 Dog     | "a large brown dog laying on top of a grass covered field"         |

![Screenshot%202025-06-23](%20091543.png)
---

## 📊 BLEU Score Performance (Top 100 Validation Samples)

| BLEU Metric | Score   |
|-------------|---------|
| BLEU-1      | 0.3541  |
| BLEU-2      | 0.2027  |
| BLEU-3      | 0.1269  |
| BLEU-4      | 0.0853  |

---

## 🧠 Model Architecture, Training, Decoding, and Deployment (Single Flow)

This model uses a CNN-based encoder and a Transformer-based decoder. The ResNet-50 encoder extracts spatial features from the image, which are passed to the Transformer decoder to generate word-by-word captions. Beam Search improves generation quality, and a Gradio app is used for deployment.

```python
# Encoder (ResNet-50)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
features = nn.Sequential(*list(resnet.children())[:-2])
features = nn.Conv2d(2048, embed_size, kernel_size=1)(features)

# Decoder (Transformer)
embedding = nn.Embedding(vocab_size, embed_size)
pos_enc = PositionalEncoding(embed_size)
decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads=4, dim_feedforward=512)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
linear = nn.Linear(embed_size, vocab_size)

# Beam Search
log_probs = torch.nn.functional.log_softmax(output[-1, 0], dim=0)
topk = torch.topk(log_probs, beam_size)

# Training
criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    for imgs, caps in dataloader:
        imgs, caps = imgs.to(device), caps.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(imgs, caps[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), caps[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch}.pth')

# Deployment
def generate_caption(image):
    tensor = transform(image).to(device)
    caption = generate_caption_beam_search(decoder, encoder, tensor, vocab)
    return caption

import gradio as gr
gr.Interface(fn=generate_caption, inputs="image", outputs="text").launch()
📦 Tech Stack
Dataset: MS COCO 2017 via torchvision.datasets.CocoCaptions

Framework: PyTorch with CUDA + AMP

Encoder: Pre-trained ResNet-50

Decoder: Multi-layer Transformer

Tokenization: Custom Vocabulary class

Loss: CrossEntropyLoss (with mask)

Optimizer: Adam

Inference: Beam Search decoding

UI: Gradio app for real-time caption generation

🧾 Project Structure
bash
Always show details

Copy
├── app.py             # Gradio app
├── model.py           # Encoder+Decoder definition
├── encoder.py         # ResNet-50 feature extractor
├── decoder.py         # Transformer decoder
├── dataset.py         # COCO loader & preprocessing
├── vocab.py           # Vocabulary & tokenization
├── train.py           # Training loop with AMP
├── utils.py           # Beam Search, BLEU eval, helpers
├── checkpoints/       # Saved weights
├── captions/          # Sample outputs
├── requirements.txt   # Dependencies
└── README.md          # This file
