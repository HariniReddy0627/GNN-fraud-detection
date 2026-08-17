import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

MAX_TRANSACTIONS = 500000
TRAIN_RATIO = 0.80
HIDDEN_CHANNELS = 64


# ============================================================
# LOAD DATA
# ============================================================

print("Loading cleaned data...")

df = pd.read_csv(
    "../data/cleaned_paysim.csv"
)

df = df.iloc[
    :MAX_TRANSACTIONS
].copy()

split_index = int(
    len(df) * TRAIN_RATIO
)

train_df = df.iloc[
    :split_index
].copy()

test_df = df.iloc[
    split_index:
].copy()


print("Training:", len(train_df))
print("Testing :", len(test_df))


# ============================================================
# ACCOUNT MAPPING
# ============================================================

accounts = pd.unique(
    pd.concat(
        [
            df["nameOrig"],
            df["nameDest"]
        ],
        ignore_index=True
    )
)

node_mapping = {
    account: i
    for i, account in enumerate(accounts)
}

num_nodes = len(node_mapping)


train_df["source"] = (
    train_df["nameOrig"].map(node_mapping)
)

train_df["destination"] = (
    train_df["nameDest"].map(node_mapping)
)

test_df["source"] = (
    test_df["nameOrig"].map(node_mapping)
)

test_df["destination"] = (
    test_df["nameDest"].map(node_mapping)
)


# ============================================================
# HISTORICAL TRAINING GRAPH
# ============================================================

train_source = train_df[
    "source"
].to_numpy(dtype=np.int64)

train_destination = train_df[
    "destination"
].to_numpy(dtype=np.int64)

train_edge_index = torch.tensor(
    np.vstack(
        [
            train_source,
            train_destination
        ]
    ),
    dtype=torch.long
)


# ============================================================
# HISTORICAL NODE FEATURES
# ============================================================

out_count = np.bincount(
    train_source,
    minlength=num_nodes
)

in_count = np.bincount(
    train_destination,
    minlength=num_nodes
)

out_amount = np.zeros(
    num_nodes,
    dtype=np.float64
)

in_amount = np.zeros(
    num_nodes,
    dtype=np.float64
)

np.add.at(
    out_amount,
    train_source,
    train_df["amount"].to_numpy()
)

np.add.at(
    in_amount,
    train_destination,
    train_df["amount"].to_numpy()
)

node_features = np.column_stack(
    [
        np.log1p(out_count),
        np.log1p(in_count),
        np.log1p(out_amount),
        np.log1p(in_amount)
    ]
)

x = torch.tensor(
    node_features,
    dtype=torch.float
)


# Normalize node features

x_mean = x.mean(
    dim=0,
    keepdim=True
)

x_std = x.std(
    dim=0,
    keepdim=True
)

x = (
    x - x_mean
) / (
    x_std + 1e-8
)


# ============================================================
# EDGE FEATURE FUNCTION
# ============================================================

def create_edge_features(data):

    amount = np.log1p(
        data["amount"].to_numpy(
            dtype=np.float64
        )
    )

    step = data["step"].to_numpy(
        dtype=np.float64
    )

    transaction_type = (
        data["type"]
        .map(
            {
                "TRANSFER": 1.0,
                "CASH_OUT": 0.0
            }
        )
        .fillna(0.0)
        .to_numpy(
            dtype=np.float64
        )
    )

    old_org = data[
        "oldbalanceOrg"
    ].to_numpy(dtype=np.float64)

    new_org = data[
        "newbalanceOrig"
    ].to_numpy(dtype=np.float64)

    old_dest = data[
        "oldbalanceDest"
    ].to_numpy(dtype=np.float64)

    new_dest = data[
        "newbalanceDest"
    ].to_numpy(dtype=np.float64)

    sender_change = old_org - new_org

    receiver_change = new_dest - old_dest

    old_org = np.log1p(
        np.maximum(old_org, 0)
    )

    new_org = np.log1p(
        np.maximum(new_org, 0)
    )

    old_dest = np.log1p(
        np.maximum(old_dest, 0)
    )

    new_dest = np.log1p(
        np.maximum(new_dest, 0)
    )

    sender_change = (
        np.sign(sender_change)
        * np.log1p(
            np.abs(sender_change)
        )
    )

    receiver_change = (
        np.sign(receiver_change)
        * np.log1p(
            np.abs(receiver_change)
        )
    )

    features = np.column_stack(
        [
            amount,
            step,
            transaction_type,
            old_org,
            new_org,
            old_dest,
            new_dest,
            sender_change,
            receiver_change
        ]
    )

    return torch.tensor(
        features,
        dtype=torch.float
    )


train_edge_attr = create_edge_features(
    train_df
)

test_edge_attr = create_edge_features(
    test_df
)


# Normalize using training statistics

edge_mean = train_edge_attr.mean(
    dim=0,
    keepdim=True
)

edge_std = train_edge_attr.std(
    dim=0,
    keepdim=True
)

train_edge_attr = (
    train_edge_attr - edge_mean
) / (
    edge_std + 1e-8
)

test_edge_attr = (
    test_edge_attr - edge_mean
) / (
    edge_std + 1e-8
)


# ============================================================
# TEST GRAPH EDGES
# ============================================================

test_source = test_df[
    "source"
].to_numpy(dtype=np.int64)

test_destination = test_df[
    "destination"
].to_numpy(dtype=np.int64)

test_edge_index = torch.tensor(
    np.vstack(
        [
            test_source,
            test_destination
        ]
    ),
    dtype=torch.long
)

test_y = torch.tensor(
    test_df["isFraud"].to_numpy(),
    dtype=torch.long
)


# ============================================================
# MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_channels=64
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            node_features,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_channels * 2
                + edge_features,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                2
            )
        )


    def encode(
        self,
        x,
        edge_index
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        return x


    def classify_edges(
        self,
        embeddings,
        edge_index,
        edge_attr
    ):

        source = edge_index[0]

        destination = edge_index[1]

        source_embedding = embeddings[
            source
        ]

        destination_embedding = embeddings[
            destination
        ]

        representation = torch.cat(
            [
                source_embedding,
                destination_embedding,
                edge_attr
            ],
            dim=1
        )

        return self.classifier(
            representation
        )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


x = x.to(device)

train_edge_index = (
    train_edge_index.to(device)
)

train_edge_attr = (
    train_edge_attr.to(device)
)

test_edge_index = (
    test_edge_index.to(device)
)

test_edge_attr = (
    test_edge_attr.to(device)
)

test_y = test_y.to(device)


# ============================================================
# LOAD BEST TEMPORAL MODEL
# ============================================================

model = GraphSAGE(
    node_features=4,
    edge_features=9,
    hidden_channels=64
).to(device)

model.load_state_dict(
    torch.load(
        "../models/temporal_gnn_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Temporal model loaded!")


# ============================================================
# GENERATE FUTURE TRANSACTION SCORES
# ============================================================

print("Generating future transaction risk scores...")

with torch.no_grad():

    embeddings = model.encode(
        x,
        train_edge_index
    )

    output = model.classify_edges(
        embeddings,
        test_edge_index,
        test_edge_attr
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )[:, 1]


scores = (
    probabilities
    .cpu()
    .numpy()
)

labels = (
    test_y
    .cpu()
    .numpy()
)


# ============================================================
# OVERALL METRICS
# ============================================================

roc_auc = roc_auc_score(
    labels,
    scores
)

pr_auc = average_precision_score(
    labels,
    scores
)

print("\n======================================")
print("TEMPORAL MODEL PERFORMANCE")
print("======================================")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)


# ============================================================
# TOP-K
# ============================================================

ranking = scores.argsort()[::-1]

total_fraud = labels.sum()

top_k_values = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000
]


print("\n======================================")
print("TEMPORAL TOP-K ANALYSIS")
print("======================================")

print(
    "\nTop-K | Fraud Found | Precision | Recall"
)

print(
    "-" * 52
)


for k in top_k_values:

    k = min(
        k,
        len(labels)
    )

    selected = ranking[:k]

    fraud_found = labels[
        selected
    ].sum()

    precision = (
        fraud_found / k
    )

    recall = (
        fraud_found / total_fraud
    )

    print(
        f"{k:5d} | "
        f"{fraud_found:11d} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f}"
    )


# ============================================================
# TOP 10
# ============================================================

print("\n======================================")
print("TOP 10 FUTURE TRANSACTIONS")
print("======================================")

for rank, index in enumerate(
    ranking[:10],
    start=1
):

    print(
        f"Rank {rank:02d} | "
        f"Score {scores[index]:.6f} | "
        f"Actual Fraud {labels[index]}"
    )