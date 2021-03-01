import os
import json

import torch
import pandas as pd
import numpy as np
import neptune
from pprint import pprint
from sklearn.metrics import r2_score
from features import features
# features = [f for f in features if f != "DR"]

from utils import encode, from_json, to_json
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--hidden_size', default=100, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--interlayer_size', default=50, type=int)
parser.add_argument('--interlayer2_size', default=0, type=int)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--dropout', default=0, type=float)

args = parser.parse_args()

neptune.init(
    project_qualified_name='agchempa/Cas13Scoring',
    api_token=
    'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOTQ0NGU4NWMtY2VhYi00ODUwLWI4ZWMtOGYxNGEwMDhlODc5In0='
)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
print(f"Using cuda? {use_cuda}")
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Load data
params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}

train = pd.read_csv("data/train.tsv", sep="\t")
val = pd.read_csv("data/val.tsv", sep="\t")

# Creates x vectors of shape (num_samples, sequence_length, feature_length)
train_dat = np.concatenate([encode(guide)
                            for guide in train.iloc[:, 0]]).astype(np.float32)
val_dat = np.concatenate([encode(guide)
                          for guide in val.iloc[:, 0]]).astype(np.float32)

# Use remaining columns as the labels to train on
train_labels = train.iloc[:, 1:].to_numpy().astype(np.float32)
val_labels = val.iloc[:, 1:].to_numpy().astype(np.float32)

output_size = train_labels.shape[-1]

print("Train data shape:", train_dat.shape)
print("Train labels shape:", train_labels.shape)

# Load numpy arrays into PyTorch DataLoader
train_data = torch.utils.data.TensorDataset(torch.from_numpy(train_dat),
                                            torch.from_numpy(train_labels))
val_data = torch.utils.data.TensorDataset(torch.from_numpy(val_dat),
                                          torch.from_numpy(val_labels))

train_loader = torch.utils.data.DataLoader(train_data, **params)
val_loader = torch.utils.data.DataLoader(val_data, **params)

# Initiate model
model_params = {
    "input_size": 6,
    "output_size": output_size,
    "hidden_size": args.hidden_size,
    "num_layers": args.num_layers,
    "interlayer_size": args.interlayer_size,
    "interlayer2_size": args.interlayer2_size,
    "bidirectional": args.bidirectional,
    "LR": args.learning_rate,
    "WD": 1e-4,
    "dropout": args.dropout,
    "name": "predict_hybridization"
}
pprint(model_params)

exp_name = model_params["name"]
model_id = hash(json.dumps(model_params, sort_keys=True))
print(f"Model ID: {model_id}")

model_name = f'models/best_vloss_{exp_name}_{model_id}.pt'
neptune.create_experiment(exp_name, params=model_params)


class ScoreNet(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 output_size,
                 interlayer_size,
                 interlayer2_size,
                 dropout=0,
                 bidirectional=False):
        super(ScoreNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.interlayer_size = interlayer_size
        self.interlayer2_size = interlayer2_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = int(bidirectional) + 1

        self.lstm = torch.nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout,
                                  batch_first=True)

        if self.interlayer2_size > 0:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.num_directions * self.hidden_size,
                                self.interlayer_size), torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dropout),
                torch.nn.Linear(self.interlayer_size, self.interlayer2_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.interlayer2_size, self.output_size))
        else:
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.num_directions * self.hidden_size,
                                self.interlayer_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.interlayer_size, self.interlayer_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.interlayer_size, self.interlayer_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.interlayer_size, self.output_size),
            )

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        output, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            # when you index into a dim with an integer, the dim disappears. Thus to stack on dim 2, you must set dim = 1
            output = torch.cat(
                (output[:, -1, :self.hidden_size], output[:, 0,
                                                          self.hidden_size:]),
                dim=1)
        else:
            output = output[:,
                            -1, :]  # output of shape (batch, seq_len, num_directions * hidden_size)
        output = output.view(batch_size, -1)

        output = self.fc(output)
        output = output.view(batch_size, -1)
        return output


model = ScoreNet(model_params["input_size"],
                 model_params["hidden_size"],
                 model_params["num_layers"],
                 output_size,
                 model_params["interlayer_size"],
                 model_params["interlayer2_size"],
                 dropout=model_params["dropout"],
                 bidirectional=model_params["bidirectional"]).to(device)

if os.path.exists(model_name):
    model.load_state_dict(torch.load(model_name, map_location=device))
print(model)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(),
                             lr=model_params["LR"],
                             weight_decay=model_params["WD"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

best_val_loss = 1e5
max_epochs = int(1e4)
for epoch in range(max_epochs):
    model.train()
    numerator, denominator = 0, 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Transfer to GPU
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        numerator += loss
        denominator += 1

        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)


        # gradient descent or adam step
        optimizer.step()
    train_loss = numerator / denominator

    # Validation
    if epoch % 1 != 0: continue

    # Validation
    dat = {feature: {"n": 0, "d": 0} for feature in features}
    numerator, denominator = 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for batch_idx, (data, targets) in enumerate(val_loader):
            # Transfer to GPU
            data = data.to(device)
            targets = targets.to(device)
            batch_size = targets.shape[0]
            # print(batch_size)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            numerator += loss
            denominator += 1

            # these have dimensions (batch_size, num_outputs)
            targets = targets.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            mse = np.power(targets - scores, 2)
            mse = np.sum(mse, axis=0).flatten()

            for idx, val in enumerate(mse):
                dat[features[idx]]["n"] += val
                dat[features[idx]]["d"] += targets.shape[0]
    val_loss = numerator / denominator
    neptune.log_metric('train_loss', train_loss)
    neptune.log_metric('validation_loss', val_loss)
    scheduler.step(val_loss)

    dat = {k: v["n"] / v["d"] for k, v in dat.items()}
    to_json(f"models/log_{model_id}.json", dat)
    for feature, loss in dat.items():
        neptune.log_metric(f'{feature}_loss', loss)

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), model_name)
        best_val_loss = val_loss

    print(f"Train loss: {train_loss}\tValidation loss: {val_loss}")
