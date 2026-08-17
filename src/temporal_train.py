import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TRANSACTIONS = 500000

TRAIN_RATIO = 0.80

EPOCHS = 15

HIDDEN_CHANNELS = 64

LEARNING_RATE = 0.001


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

print("=" * 60)
print("LOADING CLEANED DATA")
print("=" * 60)

df = pd.read_csv(
    "../data/cleaned_paysim.csv"
)

print("Total cleaned transactions:", len(df))


# Use first 500,000 transactions for this POC

df = df.iloc[
    :MAX_TRANSACTIONS
].copy()

print(
    "Transactions used:",
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


print("\nTemporal split:")

print(
    "Training transactions:",
    len(train_df)
)

print(
    "Testing transactions:",
    len(test_df)
)


print("\nTraining time range:")

print(
    train_df["step"].min(),
    "to",
    train_df["step"].max()
)


print("\nTesting time range:")

print(
    test_df["step"].min(),
    "to",
    test_df["step"].max()
)


# ============================================================
# 3. ACCOUNT NODE MAPPING
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
    account: index
    for index, account
    in enumerate(accounts)
}


num_nodes = len(node_mapping)

print(
    "Number of account nodes:",
    num_nodes
)


# ============================================================
# 4. MAP TRAINING EDGES
# ============================================================

train_df["source"] = (
    train_df["nameOrig"]
    .map(node_mapping)
)

train_df["destination"] = (
    train_df["nameDest"]
    .map(node_mapping)
)


test_df["source"] = (
    test_df["nameOrig"]
    .map(node_mapping)
)

test_df["destination"] = (
    test_df["nameDest"]
    .map(node_mapping)
)


# ============================================================
# 5. BUILD TRAINING GRAPH
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
# 6. CREATE NODE FEATURES
#    USING TRAINING DATA ONLY
# ============================================================

print("\nCreating historical node features...")


# Outgoing count

out_count = np.bincount(
    train_source,
    minlength=num_nodes
)


# Incoming count

in_count = np.bincount(
    train_destination,
    minlength=num_nodes
)


# Outgoing amount

out_amount = np.zeros(
    num_nodes,
    dtype=np.float64
)

np.add.at(
    out_amount,
    train_source,
    train_df["amount"].to_numpy()
)


# Incoming amount

in_amount = np.zeros(
    num_nodes,
    dtype=np.float64
)

np.add.at(
    in_amount,
    train_destination,
    train_df["amount"].to_numpy()
)


# Log transform

out_amount = np.log1p(
    out_amount
)

in_amount = np.log1p(
    in_amount
)


# Node feature matrix

node_features = np.column_stack(
    [
        np.log1p(out_count),
        np.log1p(in_count),
        out_amount,
        in_amount
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


print(
    "Node feature shape:",
    x.shape
)


# ============================================================
# 7. CREATE TRAIN EDGE FEATURES
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


train_edge_attr = create_edge_features(
    train_df
)

test_edge_attr = create_edge_features(
    test_df
)


# Normalize edge features using TRAINING statistics only

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


print(
    "Train edge feature shape:",
    train_edge_attr.shape
)

print(
    "Test edge feature shape:",
    test_edge_attr.shape
)


# ============================================================
# 8. TEST EDGES
# ============================================================

test_source = test_df[
    "source"
].to_numpy(
    dtype=np.int64
)

test_destination = test_df[
    "destination"
].to_numpy(
    dtype=np.int64
)


test_edge_index = torch.tensor(
    np.vstack(
        [
            test_source,
            test_destination
        ]
    ),
    dtype=torch.long
)


# ============================================================
# 9. LABELS
# ============================================================

train_y = torch.tensor(
    train_df["isFraud"].to_numpy(),
    dtype=torch.long
)

test_y = torch.tensor(
    test_df["isFraud"].to_numpy(),
    dtype=torch.long
)


print("\nClass distribution:")

print(
    "Train normal:",
    (train_y == 0).sum().item()
)

print(
    "Train fraud:",
    (train_y == 1).sum().item()
)

print(
    "Test normal:",
    (test_y == 0).sum().item()
)

print(
    "Test fraud:",
    (test_y == 1).sum().item()
)


# ============================================================
# 10. MODEL
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(
        self,
        node_features,
        edge_features,
        hidden_channels
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
        node_embeddings,
        edge_index,
        edge_attr
    ):

        source = edge_index[0]

        destination = edge_index[1]

        source_embedding = (
            node_embeddings[source]
        )

        destination_embedding = (
            node_embeddings[destination]
        )

        edge_representation = torch.cat(
            [
                source_embedding,
                destination_embedding,
                edge_attr
            ],
            dim=1
        )

        return self.classifier(
            edge_representation
        )


# ============================================================
# 11. DEVICE
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

train_edge_attr = (
    train_edge_attr.to(device)
)

train_y = train_y.to(device)

test_edge_index = (
    test_edge_index.to(device)
)

test_edge_attr = (
    test_edge_attr.to(device)
)

test_y = test_y.to(device)


# ============================================================
# 12. CREATE MODEL
# ============================================================

model = GraphSAGE(
    node_features=4,
    edge_features=9,
    hidden_channels=HIDDEN_CHANNELS
).to(device)


# ============================================================
# 13. CLASS WEIGHTS
# ============================================================

class_weights = torch.tensor(
    [
        1.0,
        15.0
    ],
    dtype=torch.float,
    device=device
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 14. TRAINING
# ============================================================

best_pr_auc = 0.0

best_epoch = 0

print("\n" + "=" * 60)
print("STARTING TEMPORAL GNN TRAINING")
print("=" * 60)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    optimizer.zero_grad()


    # ----------------------------------------
    # Build embeddings ONLY from historical
    # training graph
    # ----------------------------------------

    embeddings = model.encode(
        x,
        train_edge_index
    )


    # ----------------------------------------
    # Classify training transactions
    # ----------------------------------------

    train_output = model.classify_edges(
        embeddings,
        train_edge_index,
        train_edge_attr
    )


    loss = criterion(
        train_output,
        train_y
    )


    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


    optimizer.step()


    # ========================================================
    # EVALUATE FUTURE TRANSACTIONS
    # ========================================================

    model.eval()

    with torch.no_grad():

        embeddings = model.encode(
            x,
            train_edge_index
        )

        test_output = model.classify_edges(
            embeddings,
            test_edge_index,
            test_edge_attr
        )

        probabilities = torch.softmax(
            test_output,
            dim=1
        )[:, 1]


    y_true = (
        test_y
        .detach()
        .cpu()
        .numpy()
    )

    y_probability = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )


    # 0.5 threshold only for monitoring

    y_pred = (
        y_probability >= 0.5
    ).astype(int)


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

        pr_auc = average_precision_score(
            y_true,
            y_probability
        )

    else:

        roc_auc = 0.0

        pr_auc = 0.0


    print(
        f"Epoch {epoch:02d} | "
        f"Loss {loss.item():.4f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f} | "
        f"ROC-AUC {roc_auc:.4f} | "
        f"PR-AUC {pr_auc:.4f}"
    )


    # Save best model according to PR-AUC

    if pr_auc > best_pr_auc:

        best_pr_auc = pr_auc

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            "../models/temporal_gnn_model.pth"
        )


# ============================================================
# 15. FINAL
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL TRAINING COMPLETED")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best PR-AUC:",
    round(best_pr_auc, 4)
)

print(
    "\nSaved:"
)

print(
    "models/temporal_gnn_model.pth"
)