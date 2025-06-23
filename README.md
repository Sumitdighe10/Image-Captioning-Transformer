📊 BLEU Score Performance (Top 100 Validation Samples)

BLEU-1: 0.3541

BLEU-2: 0.2027

BLEU-3: 0.1269

BLEU-4: 0.0853

These scores reflect competitive accuracy for a lightweight model trained on limited compute.

🧰 Architecture Overview

🔢 Encoder (CNN)

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
features = nn.Sequential(*list(resnet.children())[:-2])
features = nn.Conv2d(2048, embed_size, kernel_size=1)(features)

Uses a pre-trained ResNet-50 to extract spatial features

Final fully connected layer removed

Feature maps projected to embedding size (256-dim)

Output reshaped as [sequence_len, batch_size, embed_dim]

🌐 Decoder (Transformer)

embedding = nn.Embedding(vocab_size, embed_size)
positional_encoding = PositionalEncoding(embed_size)
decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads=4, dim_feedforward=512)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

Transformer Decoder receives image embeddings as memory

Learns temporal structure using multi-head self-attention

Positional Encoding maintains sequence information

Final linear layer maps to vocab logits

🧠 Beam Search Decoding

log_probs = torch.nn.functional.log_softmax(output[-1, 0], dim=0)
topk = torch.topk(log_probs, beam_size)

Maintains multiple caption hypotheses simultaneously

Selects the highest probability caption at end of decoding

🚀 Training Pipeline

Dataset:

MS COCO 2017 captions dataset with images + 5 captions/image

Loss and Optimization:

criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

Loss computed only on non-padding tokens

Adam optimizer with gradient clipping for stability

AMP training using torch.cuda.amp for faster mixed-precision learning

Training Strategy:

50 Epochs with model checkpointing

Trained using NVIDIA GPU (CUDA enabled)

Vocabulary built from training captions using frequency threshold

🖼️ Deployment via Gradio

def generate_caption(image):
    tensor = transform(image).to(device)
    caption = generate_caption_beam_search(decoder, encoder, tensor, vocab)
    return caption

Gradio provides a drag-and-drop image upload UI

Model predicts and returns caption on submit

python app.py

Runs the app locally at http://127.0.0.1:7860

📲 Real-World Applications

🧏 Accessibility

Describe visual elements to visually impaired or deaf users via speech synthesis

🧠 AI and Computer Vision

Improve search relevance and image tagging in web platforms or e-commerce

🚗 Robotics

Enable robots to perceive and explain their environment in natural language

🔐 Surveillance

Automatically caption and log real-time camera footage for anomaly detection

⚙️ Tech Stack Breakdown (with Usage)

Component

Tool / Library

Code Integration Snippet / Use Case

Dataset

MS COCO

Used torchvision.datasets.CocoCaptions to load images + captions

Deep Learning

PyTorch

Core framework for building encoder/decoder

CNN Encoder

ResNet-50 (torchvision)

models.resnet50(weights=...) + conv projection to embedding

Decoder

nn.TransformerDecoder

Multi-layer Transformer decoder for caption generation

Text Preprocess

Custom Vocabulary class

Tokenizes, numericalizes and decodes sequences

Optimization

Adam + AMP

torch.cuda.amp and CrossEntropyLoss with mask

Caption Logic

Beam Search

Keeps top-k sequences during decoding

UI

Gradio

gr.Interface(fn=..., inputs=..., outputs=...)

GPU Acceleration

CUDA + AMP

Enabled with torch.device("cuda") and autocast()

📈 Roadmap & Enhancements

✅ Beam Search Decoding implemented

🔲 Attention Visualization heatmaps (Coming Next)

🔲 Diverse beam sampling, top-p (nucleus) sampling

🔲 Dockerization for easy cross-platform deployment

🔲 Deploy on Hugging Face Spaces / Streamlit Cloud

🔲 Caption feedback refinement using LLM
