import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x) ** 2


class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.Wf = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Uf = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))

        self.Wi = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Ui = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bi = np.zeros((hidden_dim, 1))

        self.Wo = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Uo = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bo = np.zeros((hidden_dim, 1))

        self.Wc = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Uc = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bc = np.zeros((hidden_dim, 1))

    def forward(self, x_seq):
        self.x_seq = x_seq
        T = x_seq.shape[0]
        h, c = np.zeros((self.hidden_dim, 1)), np.zeros((self.hidden_dim, 1))

        self.cache = []

        for t in range(T):
            x_t = x_seq[t].reshape(-1, 1)
            f_t = sigmoid(np.dot(self.Wf, x_t) + np.dot(self.Uf, h) + self.bf)
            i_t = sigmoid(np.dot(self.Wi, x_t) + np.dot(self.Ui, h) + self.bi)
            o_t = sigmoid(np.dot(self.Wo, x_t) + np.dot(self.Uo, h) + self.bo)
            c_tilde = tanh(np.dot(self.Wc, x_t) + np.dot(self.Uc, h) + self.bc)
            c = f_t * c + i_t * c_tilde
            h = o_t * tanh(c)
            self.cache.append((x_t, h, c, f_t, i_t, o_t, c_tilde))

        return np.array([h_t[1].flatten() for h_t in self.cache])

    def backward(self, dh_next, dc_next):
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWc = np.zeros_like(self.Wc)
        dUf = np.zeros_like(self.Uf)
        dUi = np.zeros_like(self.Ui)
        dUo = np.zeros_like(self.Uo)
        dUc = np.zeros_like(self.Uc)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)

        dh_prev = np.zeros((self.hidden_dim, 1))
        dc_prev = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(len(self.cache))):
            x_t, h, c, f_t, i_t, o_t, c_tilde = self.cache[t]
            dh = dh_next + dh_prev
            dc = dc_next + dc_prev + (dh * o_t * dtanh(c))

            do = dh * tanh(c) * dsigmoid(np.dot(self.Wo, x_t) + np.dot(self.Uo, h) + self.bo)
            di = dc * c_tilde * dsigmoid(np.dot(self.Wi, x_t) + np.dot(self.Ui, h) + self.bi)
            df = dc * c * dsigmoid(np.dot(self.Wf, x_t) + np.dot(self.Uf, h) + self.bf)
            dc_tilde = dc * i_t * dtanh(np.dot(self.Wc, x_t) + np.dot(self.Uc, h) + self.bc)

            dWf += np.dot(df, x_t.T)
            dWi += np.dot(di, x_t.T)
            dWo += np.dot(do, x_t.T)
            dWc += np.dot(dc_tilde, x_t.T)

            dUf += np.dot(df, h.T)
            dUi += np.dot(di, h.T)
            dUo += np.dot(do, h.T)
            dUc += np.dot(dc_tilde, h.T)

            dbf += df
            dbi += di
            dbo += do
            dbc += dc_tilde

            dh_prev = (np.dot(self.Uf.T, df)
                       + np.dot(self.Ui.T, di)
                       + np.dot(self.Uo.T, do)
                       + np.dot(self.Uc.T, dc_tilde))

            dc_prev = f_t * dc

        grads = {
            "Wf": dWf, "Wi": dWi, "Wo": dWo, "Wc": dWc,
            "Uf": dUf, "Ui": dUi, "Uo": dUo, "Uc": dUc,
            "bf": dbf, "bi": dbi, "bo": dbo, "bc": dbc
        }

        return grads


class BiLSTM:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001):
        self.forward_lstm = LSTMCell(input_dim, hidden_dim)
        self.backward_lstm = LSTMCell(input_dim, hidden_dim)
        self.W_out = np.random.randn(output_dim, hidden_dim * 2) * 0.01
        self.b_out = np.zeros((output_dim, 1))
        self.lr = lr

    def forward(self, x_seq):
        h_forward = self.forward_lstm.forward(x_seq)
        h_backward = self.backward_lstm.forward(np.flip(x_seq, axis=0))
        h_backward = np.flip(h_backward, axis=0)
        h_combined = np.concatenate((h_forward, h_backward), axis=1)
        logits = np.dot(h_combined, self.W_out.T) + self.b_out.T
        return logits

    def backward(self, d_logits):
        dW_out = np.dot(d_logits.T, np.concatenate((self.forward_lstm.cache[-1][1].T,
                                                    self.backward_lstm.cache[-1][1].T))) #picks the hidden states of the last time step
        db_out = np.sum(d_logits, axis=0, keepdims=True).T

        # Update output weights
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out
