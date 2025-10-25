import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3):
        super(BiLSTM, self).__init__()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # Layer normalization to stabilize training
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Projection layer from hidden -> output vocab
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to LSTM and Linear weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

        # Linear layer initialization
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x, lengths=None):
        """
        Forward pass.
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: optional tensor for packing padded sequences
        Returns:
            log_probs: (batch_size, seq_len, output_dim)
        """

        # Pack padded sequence for variable-length batching (if provided)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        # Normalize activations to prevent exploding values
        out = self.layer_norm(out)

        # Linear projection
        logits = self.fc(out)

        # Log softmax for CTC loss compatibility
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs
