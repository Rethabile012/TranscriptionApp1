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
        self.forward_lstm = LSTMCell(input_size, hidden_size)
        self.backward_lstm = LSTMCell(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, 2 * hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def get_params(self):
        # Return all trainable parameters as a list
        params = [
            self.fwd_cell.Wf, self.fwd_cell.Wi, self.fwd_cell.Wc, self.fwd_cell.Wo,
            self.fwd_cell.bf, self.fwd_cell.bi, self.fwd_cell.bc, self.fwd_cell.bo,
            self.bwd_cell.Wf, self.bwd_cell.Wi, self.bwd_cell.Wc, self.bwd_cell.Wo,
            self.bwd_cell.bf, self.bwd_cell.bi, self.bwd_cell.bc, self.bwd_cell.bo,
            self.Wy, self.by
        ]
        return params
    def get_grads(self):
        # Return gradients in same order as get_params()
        return [
            self.forward_lstm.dWf, self.forward_lstm.dWi, self.forward_lstm.dWc, self.forward_lstm.dWo,
            self.forward_lstm.dbf, self.forward_lstm.dbi, self.forward_lstm.dbc, self.forward_lstm.dbo,
            self.backward_lstm.dWf, self.backward_lstm.dWi, self.backward_lstm.dWc, self.backward_lstm.dWo,
            self.backward_lstm.dbf, self.backward_lstm.dbi, self.backward_lstm.dbc, self.backward_lstm.dbo,
            self.dWy, self.dby
        ]

    def get_weights(self):
        # For saving/loading model
        return {
            "forward_lstm_Wf": self.forward_lstm.Wf, "forward_lstm_Wi": self.forward_lstm.Wi,
            "forward_lstm_Wc": self.forward_lstm.Wc, "forward_lstm_Wo": self.forward_lstm.Wo,
            "forward_lstm_bf": self.forward_lstm.bf, "forward_lstm_bi": self.forward_lstm.bi,
            "forward_lstm_bc": self.forward_lstm.bc, "forward_lstm_bo": self.forward_lstm.bo,
            "backward_lstm_Wf": self.backward_lstm.Wf, "backward_lstm_Wi": self.backward_lstm.Wi,
            "backward_lstm_Wc": self.backward_lstm.Wc, "backward_lstm_Wo": self.backward_lstm.Wo,
            "backward_lstm_bf": self.backward_lstm.bf, "backward_lstm_bi": self.backward_lstm.bi,
            "backward_lstm_bc": self.backward_lstm.bc, "backward_lstm_bo": self.backward_lstm.bo,
            "Wy": self.Wy, "by": self.by
        }   
        
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0, keepdims=True)
    
    def forward(self, inputs):
        T = len(inputs)
        h_forward, h_backward = {}, {}
        h_prev_f, c_prev_f = np.zeros((self.forward_lstm.hidden_size, 1)), np.zeros((self.forward_lstm.hidden_size, 1))
        h_prev_b, c_prev_b = np.zeros((self.backward_lstm.hidden_size, 1)), np.zeros((self.backward_lstm.hidden_size, 1))
        
        # Forward pass
        for t in range(T):
            h_forward[t], c_prev_f = self.forward_lstm.forward(inputs[t], h_prev_f, c_prev_f)
            h_prev_f = h_forward[t]
        
        # Backward pass
        for t in reversed(range(T)):
            h_backward[t], c_prev_b = self.backward_lstm.forward(inputs[t], h_prev_b, c_prev_b)
            h_prev_b = h_backward[t]
        
        # Concatenate and apply softmax
        outputs = []
        for t in range(T):
            h_concat = np.concatenate((h_forward[t], h_backward[t]), axis=0)
            logits = np.dot(self.Wy, h_concat) + self.by
            probs = self.softmax(logits)
            outputs.append(probs)
        
        return np.array(outputs).squeeze()

    def backward(self, targets):
        T = len(self.last_outputs)
        assert T == len(targets), "targets length must match outputs length"

        # grads for Wy/by
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        loss = 0.0

        # For collecting dh contributions per time-step for fwd and bwd LSTMs
        dh_f_per_t = [np.zeros((self.hidden_size, 1)) for _ in range(T)]
        dh_b_per_t = [np.zeros((self.hidden_size, 1)) for _ in range(T)]

        for t in range(T):
            y_pred = self.last_outputs[t]  # (V,1)
            y_true = targets[t]            # (V,1)
            loss -= np.sum(y_true * np.log(y_pred + 1e-9))
            dy = y_pred - y_true           # (V,1)
            dWy += dy @ self.last_concats[t].T
            dby += dy

            # propagate into concatenated hidden
            dconcat = self.Wy.T @ dy  # (2H,1)
            dh_f = dconcat[:self.hidden_size, :]
            dh_b = dconcat[self.hidden_size:, :]

            dh_f_per_t[t] += dh_f
            dh_b_per_t[t] += dh_b

        # update Wy and by (SGD)
        # clip grads
        np.clip(dWy, -5.0, 5.0, out=dWy)
        np.clip(dby, -5.0, 5.0, out=dby)
        self.Wy -= self.lr * dWy
        self.by -= self.lr * dby

        
        self.fwd_cell.backward_through_time(dh_f_per_t, lr=self.lr)
        self.bwd_cell.backward_through_time(dh_b_per_t, lr=self.lr)

        return loss / max(1, T)
