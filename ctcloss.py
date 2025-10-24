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
    def backward(self, probs, targets, input_lengths, target_lengths):
        
        T, V = probs.shape
        L = len(targets)

        # Extend target with blanks
        extended_targets = [self.blank]
        for t in targets:
            extended_targets.extend([t, self.blank])
        S = len(extended_targets)

        # Forward (alpha) and backward (beta)
        alpha = np.full((T, S), -np.inf)
        beta = np.full((T, S), -np.inf)
        alpha[0, 0] = np.log(probs[0, self.blank])
        if S > 1:
            alpha[0, 1] = np.log(probs[0, extended_targets[1]])

        # Forward pass
        for t in range(1, T):
            for s in range(S):
                prev = alpha[t-1, s]
                if s-1 >=0:
                    prev = np.logaddexp(prev, alpha[t-1, s-1])
                if s-2 >=0 and extended_targets[s] != self.blank and extended_targets[s] != extended_targets[s-2]:
                    prev = np.logaddexp(prev, alpha[t-1, s-2])
                alpha[t, s] = prev + np.log(probs[t, extended_targets[s]])

        # Backward pass
        beta[T-1, S-1] = 0.0
        beta[T-1, S-2] = 0.0
        for t in reversed(range(T-1)):
            for s in range(S):
                curr = beta[t+1, s]
                if s+1 < S:
                    curr = np.logaddexp(curr, beta[t+1, s+1])
                if s+2 < S and extended_targets[s] != self.blank and extended_targets[s] != extended_targets[s+2]:
                    curr = np.logaddexp(curr, beta[t+1, s+2])
                beta[t, s] = curr + np.log(probs[t+1, extended_targets[s]])

        # Compute gamma = posterior probability of each label at each time
        log_gamma = np.full((T, V), -np.inf)
        for t in range(T):
            for s in range(S):
                c = extended_targets[s]
                log_gamma[t, c] = np.logaddexp(log_gamma[t, c], alpha[t, s] + beta[t, s] - alpha[-1, -1])

        gamma = np.exp(log_gamma)  # posterior probabilities

        # Gradient: probs - gamma
        dY = probs - gamma
        return dY
