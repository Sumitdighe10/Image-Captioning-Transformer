import torch
import gradio as gr
import pickle
from PIL import Image
from torchvision import transforms
from model import EncoderCNN, TransformerCaptioner, Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open(r"C:\Users\sdigh\OneDrive\Desktop\Image Captioning\vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load model
embed_size = 256
encoder = EncoderCNN(embed_size).to(device)
decoder = TransformerCaptioner(embed_size=embed_size, vocab_size=len(vocab), num_heads=4, num_layers=2).to(device)

checkpoint = torch.load(r"C:\Users\sdigh\OneDrive\Desktop\Image Captioning\caption_model_final_50epochs.pth", map_location=device)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

encoder.eval()
decoder.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Beam search decoder
def generate_caption_beam_search(decoder, encoder, image, vocab, beam_size=5, max_len=50):
    with torch.no_grad():
        memory = encoder(image.unsqueeze(0).to(device))
        if memory.dim() == 4:
            B, C, H, W = memory.shape
            memory = memory.view(B, C, H * W).permute(2, 0, 1)

        sequences = [[[], 0.0, torch.tensor([[vocab.stoi["<START>"]]], device=device)]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, input_token in sequences:
                tgt_mask = decoder.generate_square_subsequent_mask(input_token.size(0)).to(device)
                output = decoder(input_token, memory, tgt_mask)
                log_probs = torch.nn.functional.log_softmax(output[-1, 0], dim=0)
                topk = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token_id = topk.indices[i].item()
                    token_log_prob = topk.values[i].item()
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob
                    new_input = torch.cat([input_token, torch.tensor([[token_id]], device=device)], dim=0)
                    all_candidates.append((new_seq, new_score, new_input))

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        final_seq = sequences[0][0]
        if vocab.stoi["<END>"] in final_seq:
            final_seq = final_seq[:final_seq.index(vocab.stoi["<END>"])]

        return vocab.decode(final_seq)

# Gradio wrapper
def generate_caption(image):
    image = transform(image).to(device)
    return generate_caption_beam_search(decoder, encoder, image, vocab, beam_size=5)

# Gradio UI
demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🖼️ Image Caption Generator",
    description="Upload an image and get a natural language caption.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
