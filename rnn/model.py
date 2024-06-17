# import torch.nn as nn
# import torch

# # RNN Model class
# class RNNModel(nn.Module):
#     def __init__(self, input_size, num_classes, drop, skip_connections, hidden_layers, hidden_size, cell_type, **kwargs):
#         super(RNNModel, self).__init__()
        
#         self.cell_type = cell_type
#         self.hidden_layers = hidden_layers
#         self.skip_connections = skip_connections

#         # TODO: ADD batch_first=True and dropout!
#         if self.cell_type == 'lstm':
#             self.lstm = nn.LSTM(input_size,hidden_size,**kwargs)
#         elif self.cell_type == 'gru':
#             self.gru = nn.GRU(input_size,hidden_size,**kwargs)
#         else:
#             self.rnn = nn.RNN(input_size,hidden_size,**kwargs) 

#         self.dropout = nn.Dropout(drop)

#         if self.hidden_layers == 2:
#             self.dense1 = torch.nn.Linear(hidden_size,hidden_size//2)
#             self.dense2 = torch.nn.Linear(hidden_size//2, num_classes)
#         elif self.hidden_layers == 1:
#             self.dense = torch.nn.Linear(hidden_size,num_classes)

#         self.relu = torch.nn.ReLU()


#     def forward(self, x, lengths):
#         #print("x: ",x.shape)
#         packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         #print("packed_x: ",packed_x.shape)
#         if self.cell_type == 'lstm':
#             packed_out, (hidden,_) = self.lstm(packed_x)
#         elif self.cell_type == 'gru':
#             packed_out, hidden = self.gru(packed_x)
#         else:
#             packed_out, hidden = self.rnn(packed_x)
#         #print("packed_out: ",packed_out.shape)
#         # Unpack the sequence
#         out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
#         #print("out: ",out.shape)
#         # Decode the hidden state of the last time step
#         out = out[range(len(out)), lengths - 1, :]  # Get the output from the last valid time step
#         #print("out: ",out.shape)
#         if self.skip_connections and self.hidden_layers == 2:
#             dense_out_first = self.relu(self.dense1(out))
#             out_skip = self.dropout(out) + dense_out_first  # Skip connection from input to the first dense layer
#             dense_out = self.dense2(out_skip)
        
#         elif self.hidden_layers == 2:
#             out = self.dropout(out)
#             dense_out_first = self.relu(self.dense1(out))
#             dense_out = self.dense2(dense_out_first)
        
#         elif self.skip_connections and self.hidden_layers == 1:
#             out_skip = self.dropout(out) + self.dense(out)  # Skip connection directly to the output layer
#             dense_out = out_skip
        
#         else:
#             out = self.dropout(out)
#             dense_out = self.dense(out)
#         #print("dense_out: ",out.dense_out.shape)
#         return dense_out

import torch.nn as nn
import torch

# RNN Model class
class RNNModel(nn.Module):
    def __init__(self, input_size, num_classes, drop, skip_connections, hidden_layers, hidden_size, cell_type, bidirectional=False, **kwargs):
        super(RNNModel, self).__init__()
        
        self.cell_type = cell_type
        self.hidden_layers = hidden_layers
        self.skip_connections = skip_connections
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if self.cell_type == 'lstm':
            self.rnn_layer = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional, **kwargs)
        elif self.cell_type == 'gru':
            self.rnn_layer = nn.GRU(input_size, hidden_size, bidirectional=bidirectional, **kwargs)
        else:
            self.rnn_layer = nn.RNN(input_size, hidden_size, bidirectional=bidirectional, **kwargs) 

        self.dropout = nn.Dropout(drop)

        if self.hidden_layers == 2:
            self.dense1 = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
            self.dense2 = nn.Linear(hidden_size * self.num_directions, num_classes)
        elif self.hidden_layers == 1:
            self.dense = nn.Linear(hidden_size * self.num_directions, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        if self.cell_type == 'lstm':
            packed_out, (hidden, _) = self.rnn_layer(packed_x)
        elif self.cell_type == 'gru':
            packed_out, hidden = self.rnn_layer(packed_x)
        else:
            packed_out, hidden = self.rnn_layer(packed_x)

        # For bidirectional RNN, concatenate the hidden states from both directions
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        out = hidden

        if self.skip_connections and self.hidden_layers == 2:
            dense_out_first = self.relu(self.dense1(out))
            out_skip = out + dense_out_first  # Skip connection from input to the first dense layer
            dense_out = self.dense2(self.dropout(out_skip))
        elif self.hidden_layers == 2:
            out = self.dropout(out)
            dense_out_first = self.relu(self.dense1(out))
            dense_out = self.dense2(self.dropout(dense_out_first))
        else:
            out = self.dropout(out)
            dense_out = self.dense(out)

        return dense_out