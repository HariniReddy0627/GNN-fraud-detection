import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data


# ==================================================
# 1. LOAD CLEANED DATASET
# ==================================================

print("Loading cleaned dataset...")

df = pd.read_csv(
    "../data/cleaned_paysim.csv"
)

print("Dataset loaded successfully!")
print("Total transactions:", len(df))


# ==================================================
# 2. SELECT TRANSACTIONS FOR POC
# ==================================================

MAX_TRANSACTIONS = 200000

df = df.iloc[
    :MAX_TRANSACTIONS
].copy()

print(
    "Transactions used:",
    len(df)
)


# ==================================================
# 3. CREATE NODE MAPPING
# ==================================================

print("\nCreating account-node mapping...")

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
    "Number of unique accounts:",
    num_nodes
)


# ==================================================
# 4. MAP ACCOUNTS TO NODE IDs
# ==================================================

df["source"] = (
    df["nameOrig"]
    .map(node_mapping)
)

df["destination"] = (
    df["nameDest"]
    .map(node_mapping)
)


# ==================================================
# 5. CREATE EDGE INDEX
# ==================================================

source = df["source"].to_numpy(
    dtype=np.int64
)

destination = df["destination"].to_numpy(
    dtype=np.int64
)

edge_index = torch.tensor(
    np.vstack(
        [
            source,
            destination
        ]
    ),
    dtype=torch.long
)

print(
    "Edge index shape:",
    edge_index.shape
)


# ==================================================
# 6. CREATE NODE FEATURES
# ==================================================

print("\nCreating node features...")


# ------------------------------------------
# Outgoing transaction count
# ------------------------------------------

out_count = np.bincount(
    source,
    minlength=num_nodes
)


# ------------------------------------------
# Incoming transaction count
# ------------------------------------------

in_count = np.bincount(
    destination,
    minlength=num_nodes
)


# ------------------------------------------
# Outgoing transaction amount
# ------------------------------------------

out_amount = np.zeros(
    num_nodes,
    dtype=np.float64
)

np.add.at(
    out_amount,
    source,
    df["amount"].to_numpy()
)


# ------------------------------------------
# Incoming transaction amount
# ------------------------------------------

in_amount = np.zeros(
    num_nodes,
    dtype=np.float64
)

np.add.at(
    in_amount,
    destination,
    df["amount"].to_numpy()
)


# ------------------------------------------
# Log transform amounts
# ------------------------------------------

out_amount = np.log1p(
    out_amount
)

in_amount = np.log1p(
    in_amount
)


# ------------------------------------------
# Create node feature matrix
# ------------------------------------------

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


print(
    "Node feature shape:",
    x.shape
)


# ==================================================
# 7. CREATE EDGE FEATURES
# ==================================================

print("\nCreating edge features...")


# ------------------------------------------
# Amount
# ------------------------------------------

amount = np.log1p(
    df["amount"].to_numpy(
        dtype=np.float64
    )
)


# ------------------------------------------
# Time step
# ------------------------------------------

step = df["step"].to_numpy(
    dtype=np.float64
)


# ------------------------------------------
# Transaction type
#
# TRANSFER = 1
# CASH_OUT = 0
# ------------------------------------------

transaction_type = (
    df["type"]
    .map(
        {
            "TRANSFER": 1,
            "CASH_OUT": 0
        }
    )
    .to_numpy(
        dtype=np.float64
    )
)


# ------------------------------------------
# Account balances
# ------------------------------------------

old_balance_org = (
    df["oldbalanceOrg"]
    .to_numpy(dtype=np.float64)
)

new_balance_org = (
    df["newbalanceOrig"]
    .to_numpy(dtype=np.float64)
)

old_balance_dest = (
    df["oldbalanceDest"]
    .to_numpy(dtype=np.float64)
)

new_balance_dest = (
    df["newbalanceDest"]
    .to_numpy(dtype=np.float64)
)


# ------------------------------------------
# Balance changes
# ------------------------------------------

sender_balance_change = (
    old_balance_org
    - new_balance_org
)

receiver_balance_change = (
    new_balance_dest
    - old_balance_dest
)


# ------------------------------------------
# Log transform large monetary values
# ------------------------------------------

old_balance_org = np.log1p(
    np.maximum(old_balance_org, 0)
)

new_balance_org = np.log1p(
    np.maximum(new_balance_org, 0)
)

old_balance_dest = np.log1p(
    np.maximum(old_balance_dest, 0)
)

new_balance_dest = np.log1p(
    np.maximum(new_balance_dest, 0)
)

sender_balance_change = np.sign(
    sender_balance_change
) * np.log1p(
    np.abs(sender_balance_change)
)

receiver_balance_change = np.sign(
    receiver_balance_change
) * np.log1p(
    np.abs(receiver_balance_change)
)


# ------------------------------------------
# Combine edge features
# ------------------------------------------

edge_features = np.column_stack(
    [
        amount,
        step,
        transaction_type,
        old_balance_org,
        new_balance_org,
        old_balance_dest,
        new_balance_dest,
        sender_balance_change,
        receiver_balance_change
    ]
)


edge_attr = torch.tensor(
    edge_features,
    dtype=torch.float
)


print(
    "Edge feature shape:",
    edge_attr.shape
)


# ==================================================
# 8. CREATE FRAUD LABELS
# ==================================================

edge_labels = torch.tensor(
    df["isFraud"].to_numpy(),
    dtype=torch.long
)

print(
    "Fraud label shape:",
    edge_labels.shape
)


# ==================================================
# 9. SHOW CLASS DISTRIBUTION
# ==================================================

normal_count = (
    edge_labels == 0
).sum().item()

fraud_count = (
    edge_labels == 1
).sum().item()

print("\nTransaction labels:")

print(
    "Normal:",
    normal_count
)

print(
    "Fraud:",
    fraud_count
)


# ==================================================
# 10. CREATE PYTORCH GEOMETRIC GRAPH
# ==================================================

graph = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=edge_labels
)


# ==================================================
# 11. GRAPH INFORMATION
# ==================================================

print("\n======================================")
print("IMPROVED GRAPH CREATED")
print("======================================")

print(
    "Nodes:",
    graph.num_nodes
)

print(
    "Edges:",
    graph.num_edges
)

print(
    "Node features:",
    graph.x.shape
)

print(
    "Edge features:",
    graph.edge_attr.shape
)

print(
    "Edge labels:",
    graph.y.shape
)


# ==================================================
# 12. SAVE GRAPH
# ==================================================

torch.save(
    graph,
    "../data/transaction_graph.pt"
)

print(
    "\nImproved graph saved successfully!"
)

print(
    "File: data/transaction_graph.pt"
)