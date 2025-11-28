import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------
# Tokenizer
# -------------------------
def build_tokenizer(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(s, stoi):
    return [stoi[c] for c in s if c in stoi]

def decode(l, itos):
    return ''.join([itos[i] for i in l])

# -------------------------
# Tiny GPT model
# -------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        B, T, _ = x.size()
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x2, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + x2
        x = x + self.ff(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, block_size=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(idx) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.fc_out(x)
        return logits

# -------------------------
# Sampling helpers
# -------------------------
def top_k_logits(logits, k):
    if k == 0:
        return logits
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

def top_p_logits(logits, p):
    if p <= 0.0 or p > 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    mask = cumulative_probs > p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
    output_logits = torch.full_like(logits, float('-inf'))
    output_logits.scatter_(1, sorted_idx, sorted_logits)
    return output_logits

# -------------------------
# Text generation
# -------------------------
@torch.no_grad()
def generate(model, start_str, stoi, itos, max_new_tokens=200, temperature=0.7, top_k=10, top_p=0.9, sample=True, device='cpu'):
    model.eval()
    model.to(device)
    idx = torch.tensor([encode(start_str, stoi)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        if sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
    return decode(idx[0].tolist(), itos)

# -------------------------
# Helper: get batch
# -------------------------
def get_batch(data, block_size, batch_size=32, device='cpu'):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix])
    return x.to(device), y.to(device)

# -------------------------
# Main training + generation
# -------------------------
if __name__ == "__main__":
    # Load corpus
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = "To be, or not to be: that is the question.\n"

    stoi, itos = build_tokenizer(text)
    vocab_size = len(stoi)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode text
    data = encode(text, stoi)
    data_tensor = torch.tensor(data, dtype=torch.long)

    # Hyperparameters
    block_size = 64
    batch_size = 32
    epochs = 200   # increase for better results
    lr = 3e-4

    # Model
    model = TinyGPT(vocab_size=vocab_size, d_model=256, n_heads=4, n_layers=6, block_size=block_size)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Training Tiny GPT...")
    for epoch in range(epochs):
        x_batch, y_batch = get_batch(data_tensor, block_size, batch_size, device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), "tiny_gpt_trained.pt")
    print("Training complete. Model saved as tiny_gpt_trained.pt")

    # Interactive generation
    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.lower() == "quit":
            break
        out = generate(
            model,
            start_str=prompt,
            stoi=stoi,
            itos=itos,
            max_new_tokens=200,
            temperature=1.0,
            top_k=min(10, len(stoi)),
            top_p=0.9,
            sample=True,
            device=device
        )
        print("\n=== GENERATED TEXT ===\n")
        print(out)
