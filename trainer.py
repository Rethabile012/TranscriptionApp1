import numpy as np
from dataset import Dataset
from acoustic_model import BiLSTM
from decoder import CEDecoder
from ctcloss import CTCLoss

class AdamOptimizer:
    def __init__(self,params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params: dict, grads: dict):
        self.t += 1
        for key in params.keys():
            g = grads[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g * g)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class TextEncoder:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'") + ["_"]
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.blank = self.char2idx["_"]

    def text_to_indices(self, text):
        return [self.char2idx[ch] for ch in text.lower() if ch in self.char2idx]

    def indices_to_text(self, indices):
        return "".join(self.idx2char[i] for i in indices if i in self.idx2char)  # joins indices to single string


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
    # x: (T, V)
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def train_model(epochs=50, hidden_size=256, lr=0.001,
                save_path="acoustic_model_best.npz",
                history_path="training_history.npz"):

    print("Loading dataset...")
    dataset = Dataset()
    train_data = dataset.get_all_data()
    val_data = dataset.get_validation_data()
    #test_data = dataset.get_test_data()

    encoder = TextEncoder()
    decoder = CEDecoder(encoder.idx2char)
    input_size = 13
    output_size = len(encoder.chars)

    model = BiLSTM(input_size, hidden_size, output_size)
    ctc_loss_fn = CTCLoss(blank=encoder.blank)

    best_val_cer = float("inf")
    history = {"train_loss": [], "train_cer": [], "val_cer": []}

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss, total_cer, count = 0, 0, 0

        for mfcc, transcript in train_data:
            if not transcript.strip():
                continue

            # Prepare inputs
            inputs = [mfcc[:, t].reshape(-1, 1) for t in range(mfcc.shape[1])]
            outputs = model.forward(inputs)  # list of (V,1) per timestep
            y_probs = np.hstack(outputs).T   # (T, V)

            # Softmax manually
            y_probs = np.exp(y_probs - np.max(y_probs, axis=1, keepdims=True))
            y_probs /= np.sum(y_probs, axis=1, keepdims=True)

            # Encode target text
            target_indices = encoder.text_to_indices(transcript)
            if not target_indices:
                continue

            # Input and target lengths for CTC
            input_lengths = [len(inputs)]
            target_lengths = [len(target_indices)]

            # Compute CTC loss
            ctc_loss = ctc_loss_fn.forward(y_probs, target_indices, input_lengths, target_lengths)
            total_loss += ctc_loss

            # Compute CER
            pred_indices = np.argmax(y_probs, axis=1)
            pred_text = encoder.indices_to_text(pred_indices)
            total_cer += cer(transcript, pred_text)
            count += 1

        avg_loss = total_loss / max(count, 1)
        avg_cer = total_cer / max(count, 1)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}")

        # Validation
        val_cer, val_count = 0, 0
        for mfcc, transcript in val_data:
            if not transcript.strip():
                continue

            inputs = [mfcc[:, t].reshape(-1, 1) for t in range(mfcc.shape[1])]
            outputs = model.forward(inputs)
            y_probs = np.hstack(outputs).T
            y_probs = np.exp(y_probs - np.max(y_probs, axis=1, keepdims=True))
            y_probs /= np.sum(y_probs, axis=1, keepdims=True)

            pred_text = decoder.greedy_decode(y_probs)
            val_cer += cer(transcript, pred_text)
            val_count += 1

        avg_val_cer = val_cer / max(val_count, 1)
        print(f"[VAL] Epoch {epoch+1}/{epochs} - CER: {avg_val_cer:.4f}")

        history["train_loss"].append(avg_loss)
        history["train_cer"].append(avg_cer)
        history["val_cer"].append(avg_val_cer)
        np.savez(history_path, **history)

        # Save best model
        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            np.savez(save_path, **model.get_weights())
            print(f"Model improved! Saved with CER={avg_val_cer:.4f}")

    print("Training complete!")



if __name__ == "__main__":
    train_model(epochs=50, hidden_size=256, lr=0.005)
