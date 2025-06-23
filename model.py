import torch
import torch.nn as nn
import torchvision.models as models
import math

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1)

    def forward(self, images):
        features = self.resnet(images)
        features = self.conv(features)
        B, C, H, W = features.shape
        features = features.view(B, C, H * W)
        features = features.permute(2, 0, 1)  # [seq_len, batch, embed_dim]
        return features

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerCaptioner(nn.Module):
    def __init__(self, embed_size, vocab_size, num_heads, num_layers, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        decoder_layer = nn.TransformerDecoderLayer(embed_size, num_heads, 512, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

    def forward(self, tgt, memory, tgt_mask=None):
        tgt_emb = self.embed(tgt) * math.sqrt(self.embed_size)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return text.lower().translate(str.maketrans('', '', '.,!?')).split()

    def build_vocab(self, sentence_list):
        from collections import Counter
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in self.tokenizer(text)]

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.itos.get(idx, "<UNK>")
            if word in ("<START>", "<END>", "<PAD>"):
                continue
            words.append(word)
        return " ".join(words)
