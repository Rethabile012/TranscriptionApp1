import numpy as np

class CTCLoss:
    def __init__(self, blank=0):
        self.blank = blank  # usually index 0 is reserved for blank

    def forward(self, probs, targets, input_lengths, target_lengths):
        T, V = probs.shape
        L = len(targets)
        
        # Extend target with blanks
        extended_targets = [self.blank]
        for t in targets:
            extended_targets.extend([t, self.blank])
        S = len(extended_targets)

        # Initialize alpha (forward probabilities)
        alpha = np.full((T, S), -np.inf)
        alpha[0, 0] = np.log(probs[0, self.blank])
        alpha[0, 1] = np.log(probs[0, extended_targets[1]]) if S > 1 else -np.inf

        # Forward recursion
        for t in range(1, T):
            for s in range(S):
                prev = alpha[t - 1, s]

                # Stay in same state
                curr = prev

                # Move to next
                if s - 1 >= 0:
                    curr = np.logaddexp(curr, alpha[t - 1, s - 1])

                # Skip a blank or repeated char
                if s - 2 >= 0 and extended_targets[s] != self.blank and extended_targets[s] != extended_targets[s - 2]:
                    curr = np.logaddexp(curr, alpha[t - 1, s - 2])

                alpha[t, s] = curr + np.log(probs[t, extended_targets[s]])

        # Final probability (log-sum-exp of last two states)
        loss = -np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
        return loss
