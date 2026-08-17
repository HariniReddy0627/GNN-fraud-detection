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

NUMBER_OF_STREAM_TRANSACTIONS = 20

HIDDEN_CHANNELS = 64


# Demo thresholds
HIGH_THRESHOLD = 0.50
MEDIUM_THRESHOLD = 0.45


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("        GNN STREAMING FRAUD DETECTION")
print("=" * 65)

print("\nLoading PaySim data...")

df = pd.read_csv(
    "../data/cleaned_paysim.csv"
)

df = df.iloc[
    :MAX_TRANSACTIONS
].copy()

print(
    "Transactions loaded:",
    len(df)
)


# ============================================================
# 2. TEMPORAL SPLIT
# ============================================================

split_index = int(
    len(df) * TRAIN_RATIO
)

train_df = df.iloc[
    :split_index
].copy()

stream_df = df.iloc[
    split_index:
].copy()


print(
    "Historical transactions:",
    len(train_df)
)

print(
    "Streaming transactions:",
    len(stream_df)
)


# ============================================================
# 3. ACCOUNT MAPPING
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
    "Number of account nodes:",
    num_nodes
)


# ============================================================
# 4. MAP HISTORICAL TRANSACTIONS
# ============================================================

train_df["source"] = (
    train_df["nameOrig"]
    .map(node_mapping)
)

train_df["destination"] = (
    train_df["nameDest"]
    .map(node_mapping)
)


# ============================================================
# 5. CREATE HISTORICAL GRAPH
# ============================================================

train_source = train_df[
    "source"
].to_numpy(
    dtype=np.int64
)

train_destination = train_df[
    "destination"
].to_numpy(
    dtype=np.int64
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
# 6. CREATE HISTORICAL NODE FEATURES
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


# Normalize historical node features

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
# 7. EDGE FEATURE CREATION
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
        *
        np.log1p(
            np.abs(sender_change)
        )
    )

    receiver_change = (
        np.sign(receiver_change)
        *
        np.log1p(
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


# Historical edge features

train_edge_attr = create_edge_features(
    train_df
)


# Normalize using historical statistics

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


# ============================================================
# 8. GRAPH SAGE MODEL
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


    def classify_edge(
        self,
        embeddings,
        edge_index,
        edge_attr
    ):

        source = edge_index[0]

        destination = edge_index[1]

        source_embedding = (
            embeddings[source]
        )

        destination_embedding = (
            embeddings[destination]
        )

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
# 9. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nUsing device:",
    device
)


x = x.to(device)

train_edge_index = (
    train_edge_index.to(device)
)

train_edge_attr = (
    train_edge_attr.to(device)
)


# ============================================================
# 10. LOAD TRAINED MODEL
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

print(
    "Temporal GNN model loaded!"
)


# ============================================================
# 11. STREAMING FUNCTION
# ============================================================

def get_risk_level(score):

    if score >= HIGH_THRESHOLD:

        return "HIGH"

    elif score >= MEDIUM_THRESHOLD:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# 12. PROCESS STREAM
# ============================================================

print("\n")
print("=" * 65)
print("STARTING LIVE TRANSACTION STREAM")
print("=" * 65)

print(
    f"\nProcessing first "
    f"{NUMBER_OF_STREAM_TRANSACTIONS} "
    f"future transactions..."
)


# We maintain graph state

current_edge_index = train_edge_index.clone()


for stream_number in range(
    NUMBER_OF_STREAM_TRANSACTIONS
):

    transaction = stream_df.iloc[
        stream_number
    ]


    sender = transaction[
        "nameOrig"
    ]

    receiver = transaction[
        "nameDest"
    ]


    # --------------------------------------------------------
    # Check that accounts exist
    # --------------------------------------------------------

    if sender not in node_mapping:

        print(
            f"\nTransaction {stream_number + 1}: "
            "Sender not found. Skipping."
        )

        continue


    if receiver not in node_mapping:

        print(
            f"\nTransaction {stream_number + 1}: "
            "Receiver not found. Skipping."
        )

        continue


    source_id = node_mapping[
        sender
    ]

    destination_id = node_mapping[
        receiver
    ]


    # --------------------------------------------------------
    # Create transaction edge
    # --------------------------------------------------------

    transaction_df = pd.DataFrame(
        [
            transaction
        ]
    )


    edge_attr = create_edge_features(
        transaction_df
    )


    # Normalize using historical statistics

    edge_attr = (
        edge_attr - edge_mean.cpu()
    ) / (
        edge_std.cpu() + 1e-8
    )


    edge_attr = edge_attr.to(
        device
    )


    edge_index = torch.tensor(
        [
            [
                source_id
            ],
            [
                destination_id
            ]
        ],
        dtype=torch.long,
        device=device
    )


    # --------------------------------------------------------
    # GNN inference
    # --------------------------------------------------------

    with torch.no_grad():

        embeddings = model.encode(
            x,
            current_edge_index
        )

        output = model.classify_edge(
            embeddings,
            edge_index,
            edge_attr
        )

        score = torch.softmax(
            output,
            dim=1
        )[0, 1].item()


    risk_level = get_risk_level(
        score
    )


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("\n")
    print("-" * 65)

    print(
        f"TRANSACTION #{stream_number + 1}"
    )

    print("-" * 65)

    print(
        "Time step :",
        transaction["step"]
    )

    print(
        "Type      :",
        transaction["type"]
    )

    print(
        "Amount    :",
        f"{transaction['amount']:.2f}"
    )

    print(
        "Sender    :",
        sender
    )

    print(
        "Receiver  :",
        receiver
    )

    print(
        "GNN Score :",
        f"{score:.6f}"
    )

    print(
        "Risk      :",
        risk_level
    )


    if risk_level == "HIGH":

        print(
            "🚨 FRAUD ALERT"
        )

    elif risk_level == "MEDIUM":

        print(
            "⚠️ REVIEW TRANSACTION"
        )

    else:

        print(
            "✓ TRANSACTION NORMAL"
        )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print(
        "Actual PaySim label:",
        int(transaction["isFraud"])
    )


    # --------------------------------------------------------
    # UPDATE GRAPH
    # --------------------------------------------------------

    current_edge_index = torch.cat(
        [
            current_edge_index,
            edge_index
        ],
        dim=1
    )


print("\n")
print("=" * 65)
print("STREAMING DEMO COMPLETED")
print("=" * 65)

print(
    "\nProcessed transactions:",
    NUMBER_OF_STREAM_TRANSACTIONS
)