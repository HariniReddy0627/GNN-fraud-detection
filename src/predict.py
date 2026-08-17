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

# Change this number to inspect another future transaction
TEST_TRANSACTION_NUMBER = 3

HIDDEN_CHANNELS = 64


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("GNN FRAUD PREDICTION")
print("=" * 60)

print("\nLoading cleaned dataset...")

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

test_df = df.iloc[
    split_index:
].copy()


print(
    "Historical transactions:",
    len(train_df)
)

print(
    "Future transactions:",
    len(test_df)
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
    "Account nodes:",
    num_nodes
)


# ============================================================
# 4. MAP TRAINING TRANSACTIONS
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


# Normalize using historical data

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
    "Node feature shape:",
    x.shape
)


# ============================================================
# 7. EDGE FEATURE FUNCTION
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


# Create historical edge features

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
# 11. CREATE HISTORICAL NODE EMBEDDINGS
# ============================================================

print(
    "\nBuilding historical graph embeddings..."
)

with torch.no_grad():

    embeddings = model.encode(
        x,
        train_edge_index
    )


print(
    "Historical embeddings created!"
)


# ============================================================
# 12. SELECT FUTURE TRANSACTION
# ============================================================

transaction_index = (
    TEST_TRANSACTION_NUMBER - 1
)

if (
    transaction_index < 0
    or
    transaction_index >= len(test_df)
):

    raise ValueError(
        "TEST_TRANSACTION_NUMBER is outside "
        "the available test transactions."
    )


transaction = test_df.iloc[
    transaction_index
]


# ============================================================
# 13. DISPLAY TRANSACTION
# ============================================================

print("\n")
print("=" * 60)
print("TRANSACTION TO ANALYZE")
print("=" * 60)

print(
    "Transaction number:",
    TEST_TRANSACTION_NUMBER
)

print(
    "Time step:",
    transaction["step"]
)

print(
    "Type:",
    transaction["type"]
)

print(
    "Amount:",
    transaction["amount"]
)

print(
    "Sender:",
    transaction["nameOrig"]
)

print(
    "Receiver:",
    transaction["nameDest"]
)


# ============================================================
# 14. MAP SENDER / RECEIVER
# ============================================================

sender = transaction["nameOrig"]

receiver = transaction["nameDest"]


if sender not in node_mapping:

    raise ValueError(
        "Sender account not found in historical graph."
    )


if receiver not in node_mapping:

    raise ValueError(
        "Receiver account not found in historical graph."
    )


source_id = node_mapping[
    sender
]

destination_id = node_mapping[
    receiver
]


# ============================================================
# 15. CREATE ONE-TRANSACTION EDGE
# ============================================================

single_transaction = pd.DataFrame(
    [
        transaction
    ]
)


single_edge_attr = create_edge_features(
    single_transaction
)


# Normalize using historical training statistics

single_edge_attr = (
    single_edge_attr - edge_mean.cpu()
) / (
    edge_std.cpu() + 1e-8
)


single_edge_attr = single_edge_attr.to(
    device
)


single_edge_index = torch.tensor(
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


# ============================================================
# 16. RUN GNN PREDICTION
# ============================================================

print(
    "\nRunning GNN inference..."
)

with torch.no_grad():

    output = model.classify_edge(
        embeddings,
        single_edge_index,
        single_edge_attr
    )

    score = torch.softmax(
        output,
        dim=1
    )[0, 1].item()


# ============================================================
# 17. RISK LEVEL
# ============================================================

if score >= 0.50:

    risk_level = "HIGH"

elif score >= 0.45:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# 18. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("GNN FRAUD PREDICTION RESULT")
print("=" * 60)

print(
    f"Transaction type : {transaction['type']}"
)

print(
    f"Amount           : {transaction['amount']:.2f}"
)

print(
    f"Sender           : {transaction['nameOrig']}"
)

print(
    f"Receiver         : {transaction['nameDest']}"
)

print(
    f"GNN risk score   : {score:.6f}"
)

print(
    f"Risk level       : {risk_level}"
)


if risk_level == "HIGH":

    print(
        "\n🚨 HIGH-RISK TRANSACTION"
    )

elif risk_level == "MEDIUM":

    print(
        "\n⚠️ MEDIUM-RISK TRANSACTION"
    )

else:

    print(
        "\n✓ LOW-RISK TRANSACTION"
    )


# ============================================================
# 19. DEMO-ONLY GROUND TRUTH
# ============================================================

print(
    "\nActual PaySim label:",
    int(transaction["isFraud"])
)

print(
    "(Ground truth is shown only for POC evaluation.)"
)

print(
    "\nPrediction completed."
)