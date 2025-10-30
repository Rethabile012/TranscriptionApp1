import numpy as np
from dataset import Dataset
from acoustic_model import BiLSTM  
from decoder import CEDecoder

class TextEncoder:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'") + ["_"]
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.blank = self.char2idx["_"]

    def text_to_indices(self, text):
        return [self.char2idx[ch] for ch in text.lower() if ch in self.char2idx]

    def indices_to_text(self, indices):
        return "".join(self.idx2char[i] for i in indices if i in self.idx2char)


def cer(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / max(m, 1)


def softmax_rows(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    s = np.sum(e, axis=1, keepdims=True)
    s[s == 0] = 1e-8
    return e / s


def train_model(epochs=5, hidden_size=128, lr=0.0005,
                save_path="bilstm_model_best.npz",
                history_path="bilstm_history.npz"):

    print("Loading dataset...")
    dataset = Dataset()
    train_data = dataset.get_all_data()
    val_data = dataset.get_validation_data()

    encoder = TextEncoder()
    decoder = CEDecoder(encoder.idx2char)
    input_size = 13  # e.g., MFCC feature dim
    output_size = len(encoder.chars)

    model = BiLSTM(input_size, hidden_size, output_size, lr=lr)
    

    # Collect model parameters for optimizer
    params = {
        "W_out": model.W_out,
        "b_out": model.b_out,
        **{f"fw_{k}": v for k, v in vars(model.forward_lstm).items() if isinstance(v, np.ndarray)},
        **{f"bw_{k}": v for k, v in vars(model.backward_lstm).items() if isinstance(v, np.ndarray)},
    }

    

    best_val_cer = float("inf")
    history = {"train_loss": [], "train_cer": [], "val_cer": []}

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss, total_cer, count = 0, 0, 0

        for mfcc, transcript in train_data:
            if not transcript.strip():
                continue

            inputs = mfcc.T.astype(np.float32)  # shape (seq_len, input_dim)
            outputs = model.forward(inputs)
            y_probs = softmax_rows(outputs)

            target_indices = encoder.text_to_indices(transcript)
            if not target_indices:
                continue

            input_lengths = [len(inputs)]
            target_lengths = [len(target_indices)]

            
            
            
            d_logits = y_probs - np.eye(y_probs.shape[1])[np.array(target_indices[:y_probs.shape[0]])]
            grads = model.backward(d_logits)


            
            pred_indices = np.argmax(y_probs, axis=1)
            pred_text = encoder.indices_to_text(pred_indices)
            total_cer += cer(transcript, pred_text)
            count += 1

        avg_loss = total_loss / max(count, 1)
        avg_cer = total_cer / max(count, 1)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}")

        # Validation
        val_cer_total, val_count = 0, 0
        for mfcc, transcript in val_data:
            if not transcript.strip():
                continue

            inputs = mfcc.T.astype(np.float32)
            outputs = model.forward(inputs)
            y_probs = softmax_rows(outputs)
            pred_text = decoder.greedy_decode(y_probs)
            val_cer_total += cer(transcript, pred_text)
            val_count += 1

        avg_val_cer = val_cer_total / max(val_count, 1)
        print(f"[VAL] Epoch {epoch+1}/{epochs} - CER: {avg_val_cer:.4f}")

        history["train_loss"].append(avg_loss)
        history["train_cer"].append(avg_cer)
        history["val_cer"].append(avg_val_cer)
        np.savez(history_path, **history)

        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            np.savez(save_path, **params)
            print(f"Model improved! Saved with CER={avg_val_cer:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train_model(epochs=5, hidden_size=128, lr=0.0005)
