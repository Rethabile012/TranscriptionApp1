# acoustic_model.py
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # weights: (hidden, hidden+input)
        limit = 0.1
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * limit
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * limit
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * limit
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * limit

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

        # gradients accumulators (reset per sequence or batch)
        self.reset_grads()

        # caches per timestep (for BPTT)
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
        concat = np.vstack((h_prev, x))  # (hidden+input, 1)

        f_pre = self.Wf @ concat + self.bf
        i_pre = self.Wi @ concat + self.bi
        c_pre = self.Wc @ concat + self.bc
        o_pre = self.Wo @ concat + self.bo

        f = sigmoid(f_pre)
        i = sigmoid(i_pre)
        c_bar = np.tanh(c_pre)
        c = f * c_prev + i * c_bar
        o = sigmoid(o_pre)
        h = o * np.tanh(c)

        if store_cache:
            cache = {
                "concat": concat,
                "h_prev": h_prev,
                "c_prev": c_prev,
                "f": f, "i": i, "c_bar": c_bar, "c": c,
                "o": o, "f_pre": f_pre, "i_pre": i_pre, "c_pre": c_pre, "o_pre": o_pre
            }
            self.caches.append(cache)

        return h, c

    def backward_through_time(self, dhs, lr=1e-3, clip=5.0):
        # initialize
        T = len(self.caches)
        # ensure dhs length matches
        if len(dhs) != T:
            # pad with zeros if needed
            if len(dhs) < T:
                dhs = dhs + [np.zeros((self.hidden_size, 1)) for _ in range(T - len(dhs))]
            else:
                dhs = dhs[:T]

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))
        # reset grads
        self.reset_grads()

        for t in reversed(range(T)):
            cache = self.caches[t]
            concat = cache["concat"]            # (H+I, 1)
            h_prev = cache["h_prev"]
            c_prev = cache["c_prev"]
            f = cache["f"]; i = cache["i"]; c_bar = cache["c_bar"]; c = cache["c"]; o = cache["o"]

            # total dh
            dh = dhs[t] + dh_next  # upstream plus next timestep's gradient

            # backprop through output
            do = dh * np.tanh(c)
            do_pre = do * o * (1 - o)

            # backprop through cell state
            dc = dh * o * (1 - np.tanh(c) ** 2) + dc_next

            di = dc * c_bar
            di_pre = di * i * (1 - i)

            dc_bar = dc * i
            dc_bar_pre = dc_bar * (1 - c_bar ** 2)

            df = dc * c_prev
            df_pre = df * f * (1 - f)

            # accumulate gradients for weights and biases
            self.dWf += df_pre @ concat.T
            self.dWi += di_pre @ concat.T
            self.dWc += dc_bar_pre @ concat.T
            self.dWo += do_pre @ concat.T

            self.dbf += df_pre
            self.dbi += di_pre
            self.dbc += dc_bar_pre
            self.dbo += do_pre

            # propagate to concat (hidden_prev + x)
            dconcat = (self.Wf.T @ df_pre) + (self.Wi.T @ di_pre) + (self.Wc.T @ dc_bar_pre) + (self.Wo.T @ do_pre)
            dh_next = dconcat[:self.hidden_size, :]
            dx = dconcat[self.hidden_size:, :]
            # propagate cell state gradient
            dc_next = dc * f

        # clip gradients
        for g in [self.dWf, self.dWi, self.dWc, self.dWo, self.dbf, self.dbi, self.dbc, self.dbo]:
            np.clip(g, -clip, clip, out=g)

        # simple SGD update (use lr param)
        self.Wf -= lr * self.dWf
        self.Wi -= lr * self.dWi
        self.Wc -= lr * self.dWc
        self.Wo -= lr * self.dWo
        self.bf -= lr * self.dbf
        self.bi -= lr * self.dbi
        self.bc -= lr * self.dbc
        self.bo -= lr * self.dbo

        # clear caches after backward
        self.caches = []

        # return last dh_next to allow stacking layers 
        return dh_next, dc_next

class BiLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size

        # forward and backward LSTM cells
        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)

        self.Why = np.random.randn(output_size, 2*hidden_size) * 0.1
        self.by  = np.zeros((output_size, 1))

        self.last_concats = []

    def forward(self, inputs):
        T = len(inputs)
        h_f_prev = np.zeros((self.hidden_size,1))
        c_f_prev = np.zeros((self.hidden_size,1))
        h_b_next = np.zeros((self.hidden_size,1))
        c_b_next = np.zeros((self.hidden_size,1))

        h_forward_seq = []
        h_backward_seq = []

        # forward LSTM
        for t in range(T):
            h_f, c_f = self.forward_lstm.forward(inputs[t], h_f_prev, c_f_prev)
            h_forward_seq.append(h_f)
            h_f_prev, c_f_prev = h_f, c_f

        # backward LSTM
        for t in reversed(range(T)):
            h_b, c_b = self.backward_lstm.forward(inputs[t], h_b_next, c_b_next)
            h_backward_seq.insert(0, h_b)
            h_b_next, c_b_next = h_b, c_b

        outputs = []
        self.last_concats = []

        for t in range(T):
            h = np.vstack([h_forward_seq[t], h_backward_seq[t]])  # (2*hidden_size,1)
            self.last_concats.append(h)
            y = self.Why @ h + self.by  # (V,1)
            outputs.append(y)

        return outputs

    def backward(self, dY_list):
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dh_forward_seq = [np.zeros((self.hidden_size,1)) for _ in range(len(dY_list))]
        dh_backward_seq = [np.zeros((self.hidden_size,1)) for _ in range(len(dY_list))]

        for t in reversed(range(len(self.last_concats))):
            h = self.last_concats[t]
            dy = dY_list[t]

            dWhy += dy @ h.T
            dby  += dy

            dh = self.Why.T @ dy  # split for forward/backward
            dh_forward_seq[t] += dh[:self.hidden_size]
            dh_backward_seq[t] += dh[self.hidden_size:]

        # propagate through LSTM cells
        self.forward_lstm.backward_through_time(dh_forward_seq)
        self.backward_lstm.backward_through_time(dh_backward_seq)

        return {"Why": dWhy, "by": dby}
