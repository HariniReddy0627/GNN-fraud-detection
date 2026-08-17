import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TRANSACTIONS = 500000
TRAIN_RATIO = 0.80

TOP_K = 20

HIDDEN_CHANNELS = 64


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("             GNN FRAUD DETECTION DEMO")
print("=" * 70)

print("\nLoading PaySim dataset...")

df = pd.read_csv(
    "../data/cleaned_paysim.csv"
)

df = df.iloc[:MAX_TRANSACTIONS].copy()

split_index = int(
    len(df) * TRAIN_RATIO
)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print(
    "Historical transactions:",
    len(train_df)
)

print(
    "Future transactions:",
    len(test_df)
)


# ============================================================
# ACCOUNT MAPPING
# ============================================================

print("\nCreating account mapping...")

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

print(
    "Account nodes:",
    num_nodes
)


# ============================================================
# HISTORICAL GRAPH
# ============================================================

train_source = (
    train_df["nameOrig"]
    .map(node_mapping)
    .to_numpy(dtype=np.int64)
)

train_destination = (
    train_df["nameDest"]
    .map(node_mapping)
    .to_numpy(dtype=np.int64)
)

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
# NODE FEATURES
# ============================================================

print("\nCreating historical node features...")

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

print(
    "Node features:",
    x.shape
)


# ============================================================
# EDGE FEATURES
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
    ].to_numpy(
        dtype=np.float64
    )

    new_org = data[
        "newbalanceOrig"
    ].to_numpy(
        dtype=np.float64
    )

    old_dest = data[
        "oldbalanceDest"
    ].to_numpy(
        dtype=np.float64
    )

    new_dest = data[
        "newbalanceDest"
    ].to_numpy(
        dtype=np.float64
    )

    sender_change = (
        old_org - new_org
    )

    receiver_change = (
        new_dest - old_dest
    )

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

    return torch.tensor(
        np.column_stack(
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
        ),
        dtype=torch.float
    )


train_edge_attr = create_edge_features(
    train_df
)

edge_mean = train_edge_attr.mean(
    dim=0,
    keepdim=True
)

edge_std = train_edge_attr.std(
    dim=0,
    keepdim=True
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

print(
    "\nDevice:",
    device
)


x = x.to(device)

train_edge_index = (
    train_edge_index.to(device)
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = GraphSAGE(
    node_features=4,
    edge_features=9,
    hidden_channels=HIDDEN_CHANNELS
).to(device)

model.load_state_dict(
    torch.load(
        "../models/temporal_gnn_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print(
    "Temporal GNN model loaded!"
)


# ============================================================
# CREATE HISTORICAL EMBEDDINGS
# ============================================================

print(
    "Building historical graph embeddings..."
)

with torch.no_grad():

    embeddings = model.encode(
        x,
        train_edge_index
    )


print(
    "Historical embeddings ready!"
)


# ============================================================
# CREATE FUTURE EDGE INDEX
# ============================================================

print(
    "Preparing future transactions..."
)

test_source = (
    test_df["nameOrig"]
    .map(node_mapping)
    .to_numpy(dtype=np.int64)
)

test_destination = (
    test_df["nameDest"]
    .map(node_mapping)
    .to_numpy(dtype=np.int64)
)

test_edge_index = torch.tensor(
    np.vstack(
        [
            test_source,
            test_destination
        ]
    ),
    dtype=torch.long,
    device=device
)


# ============================================================
# CREATE FUTURE EDGE FEATURES
# ============================================================

test_edge_attr = create_edge_features(
    test_df
)

test_edge_attr = (
    test_edge_attr - edge_mean
) / (
    edge_std + 1e-8
)

test_edge_attr = test_edge_attr.to(
    device
)


# ============================================================
# SCORE FUTURE TRANSACTIONS
# ============================================================

print(
    "Running GNN inference..."
)

with torch.no_grad():

    output = model.classify_edges(
        embeddings,
        test_edge_index,
        test_edge_attr
    )

    scores = torch.softmax(
        output,
        dim=1
    )[:, 1]


scores = (
    scores
    .cpu()
    .numpy()
)


# ============================================================
# RANK TRANSACTIONS
# ============================================================

ranking = np.argsort(
    scores
)[::-1]

top_indices = ranking[:TOP_K]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("              HIGHEST-RISK TRANSACTIONS")
print("=" * 70)

print(
    "\nRank | Risk Score | Type     | Amount       | Actual"
)

print(
    "-" * 65
)


for rank, index in enumerate(
    top_indices,
    start=1
):

    transaction = test_df.iloc[
        index
    ]

    score = scores[
        index
    ]

    actual = int(
        transaction["isFraud"]
    )

    print(
        f"{rank:4d} | "
        f"{score:.6f}   | "
        f"{transaction['type']:8s} | "
        f"{transaction['amount']:12.2f} | "
        f"{actual}"
    )


# ============================================================
# TOP-K SUMMARY
# ============================================================

labels = test_df[
    "isFraud"
].to_numpy()

total_fraud = labels.sum()


print("\n")
print("=" * 70)
print("                    MODEL SUMMARY")
print("=" * 70)


for k in [
    100,
    500,
    1000
]:

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
        f"Top {k:4d}: "
        f"{fraud_found:4d} fraud | "
        f"Precision: {precision:.2%} | "
        f"Recall: {recall:.2%}"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("                 CEO DEMO CONCLUSION")
print("=" * 70)

print(
    "\nThe GNN represents accounts as graph nodes "
    "and financial transactions as directed edges."
)

print(
    "Historical transaction relationships are used "
    "to rank future transactions by fraud risk."
)

print(
    "The highest-risk transactions can be prioritized "
    "for analyst investigation."
)

print("=" * 70)