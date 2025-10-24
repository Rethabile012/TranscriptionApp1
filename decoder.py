import numpy as np
class CEDecoder:
    def __init__(self, idx2char, beam_width=5, alpha=0.5, blank_idx=36):
        self.idx2char = idx2char
        self.beam_width = beam_width
        self.alpha = alpha
        self.blank_idx = blank_idx

    def greedy_decode(self, probs):
        best_path = np.argmax(probs, axis=1)
        decoded = []
        prev = None
        for idx in best_path:
            if idx != self.blank_idx and idx != prev:
                decoded.append(self.idx2char[idx])
            prev = idx
        return "".join(decoded)
