import torch.nn as nn
import torch
import torch.nn.functional as F

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
    
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=4, padding=200)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=10, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=10, stride=2, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=10, stride=2, padding=5)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.fc = nn.Linear(512, num_classes)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.mean(dim=-1)  # Global Average Pooling (GAP)
        x = self.fc(x)
        return x#F.log_softmax(x, dim=-1)
    
# Wrapper Model for Summary
class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, x):
        # Create a dummy additional input tensor
        additional_input = torch.Tensor([216])
        return self.model(x, additional_input)