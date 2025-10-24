import numpy as np
from dataset import Dataset
from acoustic_model import BiLSTM
from decoder import CEDecoder
from ctcloss import CTCLoss

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
    test_data = dataset.get_test_data()

    encoder = TextEncoder()
    decoder = CEDecoder(encoder.idx2char, beam_width=5, alpha=0.5)
    input_size = 13
    output_size = len(encoder.chars)

    model = BiLSTM(input_size, hidden_size, output_size)
    ctc_loss_fn = CTCLoss(blank=encoder.blank)
    optimizer = AdamOptimizer(lr=lr)
    best_val_cer = float("inf")

    history = {"train_loss": [], "train_cer": [],
               "val_cer": [], "val_wer": []}

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss, total_cer, total_wer = 0.0, 0.0, 0.0
        count = 0

        for mfcc, transcript in train_data:
            if not transcript.strip():
                continue

            # Prepare inputs as list of column vectors
            T = mfcc.shape[1]
            if T == 0:
                continue
            inputs = [mfcc[:, t].reshape(-1, 1) for t in range(T)]

            outputs = model.forward(inputs)  # expects list of (V,1) or array; keep compatibility
            # outputs may be list of (V,1) -> form (T, V)
            if isinstance(outputs, list):
                y_raw = np.hstack(outputs).T   # (T, V)
            else:
                y_raw = np.array(outputs)      # already (T, V)

            target_indices = encoder.text_to_indices(transcript)
            if not target_indices:
                continue

            # Softmax per time-step
            y_probs = softmax_rows(y_raw)  # (T, V)

            # Compute CTC loss
            ctc_loss = ctc_loss_fn.forward(y_probs, target_indices)
            total_loss += ctc_loss

            model.backward_ctc(y_probs, target_indices)

            # Update parameters via Adam (expects model.params and model.grads dicts)
            if hasattr(model, "params") and hasattr(model, "grads"):
                optimizer.update(model.params, model.grads)
            else:
                # Fallback: if model exposes get_weights() and get_grads(), use them
                if hasattr(model, "get_weights") and hasattr(model, "get_grads"):
                    params = model.get_weights()
                    grads = model.get_grads()
                    optimizer.update(params, grads)
                    # If model expects weights to be written back, user should implement that mapping
                else:
                    # If no grad API, skip optimizer update (warn)
                    print("Warning: model.params/model.grads not found — optimizer update skipped for this step.")
            
            # For quick metrics use greedy argmax decode (then collapse blanks in decoder if needed)
            pred_indices = np.argmax(y_probs, axis=1)
            pred_text = encoder.indices_to_text(pred_indices)

            total_cer += cer(transcript, pred_text)
            count += 1

        avg_loss = total_loss / max(count, 1)
        avg_cer = total_cer / max(count, 1)
        avg_wer = total_wer / max(count, 1)

        print(f"[TRAIN] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, CER: {avg_cer:.4f}")

        # Validation loop
        val_cer, val_wer, val_count = 0.0, 0.0, 0
        for mfcc, transcript in val_data:
            if not transcript.strip():
                continue

            T = mfcc.shape[1]
            if T == 0:
                continue
            inputs = [mfcc[:, t].reshape(-1, 1) for t in range(T)]
            outputs = model.forward(inputs)
            if isinstance(outputs, list):
                y_raw = np.hstack(outputs).T
            else:
                y_raw = np.array(outputs)
            y_probs = softmax_rows(y_raw)

            pred_text = decoder.beam_search(y_probs)
            val_cer += cer(transcript, pred_text)
            val_count += 1

        avg_val_cer = val_cer / max(val_count, 1)
        avg_val_wer = val_wer / max(val_count, 1)
        print(f"[VAL] Epoch {epoch+1}/{epochs} - CER: {avg_val_cer:.4f}")

        history["train_loss"].append(avg_loss)
        history["train_cer"].append(avg_cer)
        history["val_cer"].append(avg_val_cer)
        np.savez(history_path, **history)

        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            # model.get_weights() expected to return dict of numpy arrays
            if hasattr(model, "get_weights"):
                np.savez(save_path, **model.get_weights())
            print(f"Model improved! Saved with CER={avg_val_cer:.4f}")

    # Test evaluation
    test_cer, test_wer, test_count = 0.0, 0.0, 0
    for mfcc, transcript in test_data:
        if not transcript.strip():
            continue

        T = mfcc.shape[1]
        if T == 0:
            continue
        inputs = [mfcc[:, t].reshape(-1, 1) for t in range(T)]
        outputs = model.forward(inputs)
        if isinstance(outputs, list):
            y_raw = np.hstack(outputs).T
        else:
            y_raw = np.array(outputs)
        y_probs = softmax_rows(y_raw)
        pred_text = decoder.greedy_decode(y_probs)

        test_cer += cer(transcript, pred_text)

        test_count += 1

    avg_test_cer = test_cer / max(test_count, 1)
    print(f"\n[TEST] CER: {avg_test_cer:.4f}")


if __name__ == "__main__":
    train_model(epochs=50, hidden_size=256, lr=0.005)
