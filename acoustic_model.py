import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        limit = 0.1
        dtype = np.float32  # use float32 to save memory

        # weights and biases
        self.Wf = (np.random.randn(hidden_size, input_size + hidden_size) * limit).astype(dtype)
        self.Wi = (np.random.randn(hidden_size, input_size + hidden_size) * limit).astype(dtype)
        self.Wc = (np.random.randn(hidden_size, input_size + hidden_size) * limit).astype(dtype)
        self.Wo = (np.random.randn(hidden_size, input_size + hidden_size) * limit).astype(dtype)

        self.bf = np.zeros((hidden_size, 1), dtype=dtype)
        self.bi = np.zeros((hidden_size, 1), dtype=dtype)
        self.bc = np.zeros((hidden_size, 1), dtype=dtype)
        self.bo = np.zeros((hidden_size, 1), dtype=dtype)

        self.reset_grads()
        self.caches = []

    def reset_grads(self):
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWc = np.zeros_like(self.Wc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbc = np.zeros_like(self.bc)
        self.dbo = np.zeros_like(self.bo)

    def forward(self, x, h_prev, c_prev, store_cache=True):
        concat = np.vstack((h_prev, x)).astype(np.float32)

        f = sigmoid(self.Wf @ concat + self.bf)
        i = sigmoid(self.Wi @ concat + self.bi)
        c_bar = np.tanh(self.Wc @ concat + self.bc)
        c = f * c_prev + i * c_bar
        o = sigmoid(self.Wo @ concat + self.bo)
        h = o * np.tanh(c)

        if store_cache:
            # minimal cache for memory efficiency
            self.caches.append((concat, c_prev, f, i, c_bar, c, o))

        return h, c

    def backward_through_time(self, dhs, lr=1e-3, clip=5.0):
        T = len(self.caches)
        if len(dhs) != T:
            if len(dhs) < T:
                dhs = dhs + [np.zeros((self.hidden_size, 1), dtype=np.float32) for _ in range(T - len(dhs))]
            else:
                dhs = dhs[:T]

        dh_next = np.zeros((self.hidden_size, 1), dtype=np.float32)
        dc_next = np.zeros((self.hidden_size, 1), dtype=np.float32)
        self.reset_grads()

        for t in reversed(range(T)):
            concat, c_prev, f, i, c_bar, c, o = self.caches[t]
            dh = dhs[t] + dh_next
            do = dh * np.tanh(c)
            do_pre = do * o * (1 - o)
            dc = dh * o * (1 - np.tanh(c) ** 2) + dc_next
            di = dc * c_bar
            di_pre = di * i * (1 - i)
            dc_bar = dc * i
            dc_bar_pre = dc_bar * (1 - c_bar ** 2)
            df = dc * c_prev
            df_pre = df * f * (1 - f)

            # weight grads
            self.dWf += df_pre @ concat.T
            self.dWi += di_pre @ concat.T
            self.dWc += dc_bar_pre @ concat.T
            self.dWo += do_pre @ concat.T
            self.dbf += df_pre
            self.dbi += di_pre
            self.dbc += dc_bar_pre
            self.dbo += do_pre

            dconcat = (
                self.Wf.T @ df_pre + self.Wi.T @ di_pre + self.Wc.T @ dc_bar_pre + self.Wo.T @ do_pre
            )
            dh_next = dconcat[:self.hidden_size, :]
            dc_next = dc * f

        # clip and update
        for g in [self.dWf, self.dWi, self.dWc, self.dWo, self.dbf, self.dbi, self.dbc, self.dbo]:
            np.clip(g, -clip, clip, out=g)

        self.Wf -= lr * self.dWf
        self.Wi -= lr * self.dWi
        self.Wc -= lr * self.dWc
        self.Wo -= lr * self.dWo
        self.bf -= lr * self.dbf
        self.bi -= lr * self.dbi
        self.bc -= lr * self.dbc
        self.bo -= lr * self.dbo

        # clear caches to free memory
        self.caches.clear()

        return dh_next, dc_next


class BiLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)

        dtype = np.float32
        self.Why = (np.random.randn(output_size, 2 * hidden_size) * 0.1).astype(dtype)
        self.by = np.zeros((output_size, 1), dtype=dtype)

        self.last_concats = []

    def get_params(self):
        return {
            "Why": self.Why,
            "by": self.by,
            "forward_Wf": self.forward_lstm.Wf,
            "forward_Wi": self.forward_lstm.Wi,
            "forward_Wc": self.forward_lstm.Wc,
            "forward_Wo": self.forward_lstm.Wo,
            "forward_bf": self.forward_lstm.bf,
            "forward_bi": self.forward_lstm.bi,
            "forward_bc": self.forward_lstm.bc,
            "forward_bo": self.forward_lstm.bo,
            "backward_Wf": self.backward_lstm.Wf,
            "backward_Wi": self.backward_lstm.Wi,
            "backward_Wc": self.backward_lstm.Wc,
            "backward_Wo": self.backward_lstm.Wo,
            "backward_bf": self.backward_lstm.bf,
            "backward_bi": self.backward_lstm.bi,
            "backward_bc": self.backward_lstm.bc,
            "backward_bo": self.backward_lstm.bo
        }

    def get_weights(self):
        return self.get_params()

    def forward(self, inputs):
        T = len(inputs)
        h_f_prev = np.zeros((self.hidden_size, 1), dtype=np.float32)
        c_f_prev = np.zeros((self.hidden_size, 1), dtype=np.float32)
        h_b_next = np.zeros((self.hidden_size, 1), dtype=np.float32)
        c_b_next = np.zeros((self.hidden_size, 1), dtype=np.float32)

        h_forward_seq, h_backward_seq = [], []

        for t in range(T):
            h_f, c_f = self.forward_lstm.forward(inputs[t], h_f_prev, c_f_prev)
            h_forward_seq.append(h_f)
            h_f_prev, c_f_prev = h_f, c_f

        for t in reversed(range(T)):
            h_b, c_b = self.backward_lstm.forward(inputs[t], h_b_next, c_b_next)
            h_backward_seq.insert(0, h_b)
            h_b_next, c_b_next = h_b, c_b

        outputs = []
        self.last_concats = []

        for t in range(T):
            h = np.vstack([h_forward_seq[t], h_backward_seq[t]]).astype(np.float32)
            self.last_concats.append(h)
            y = self.Why @ h + self.by
            outputs.append(y)

        return outputs

    def clear_caches(self):
        """Free memory by clearing LSTM caches."""
        self.forward_lstm.caches.clear()
        self.backward_lstm.caches.clear()
