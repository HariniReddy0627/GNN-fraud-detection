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


# ==================================================
# 1. LOAD GRAPH
# ==================================================

print("Loading improved graph...")

graph = torch.load(
    "../data/transaction_graph.pt",
    weights_only=False
)

print("Graph loaded!")

print("Nodes:", graph.num_nodes)

print("Edges:", graph.num_edges)

print(
    "Node features:",
    graph.x.shape
)

print(
    "Edge features:",
    graph.edge_attr.shape
)


# ==================================================
# 2. DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)

graph = graph.to(device)


# ==================================================
# 3. NORMALIZE FEATURES
# ==================================================

# Normalize node features

node_mean = graph.x.mean(
    dim=0,
    keepdim=True
)

node_std = graph.x.std(
    dim=0,
    keepdim=True
)

graph.x = (
    graph.x - node_mean
) / (
    node_std + 1e-8
)


# Normalize edge features

edge_mean = graph.edge_attr.mean(
    dim=0,
    keepdim=True
)

edge_std = graph.edge_attr.std(
    dim=0,
    keepdim=True
)

graph.edge_attr = (
    graph.edge_attr - edge_mean
) / (
    edge_std + 1e-8
)


# ==================================================
# 4. MODEL
# ==================================================

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

            nn.Dropout(0.25),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                2
            )
        )


    def forward(
        self,
        x,
        edge_index,
        edge_attr
    ):

        # First GraphSAGE layer

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)


        # Second GraphSAGE layer

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)


        # Source and destination

        source = edge_index[0]

        destination = edge_index[1]


        source_embedding = x[source]

        destination_embedding = x[destination]


        # Edge representation

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


# ==================================================
# 5. CREATE MODEL
# ==================================================

model = GraphSAGE(
    node_features=graph.x.shape[1],
    edge_features=graph.edge_attr.shape[1],
    hidden_channels=64
)

model = model.to(device)

print(
    "\nModel created successfully!"
)


# ==================================================
# 6. TRAIN / TEST SPLIT
# ==================================================

torch.manual_seed(42)

indices = torch.randperm(
    graph.num_edges,
    device=device
)

train_size = int(
    0.8 * graph.num_edges
)

train_indices = indices[
    :train_size
]

test_indices = indices[
    train_size:
]


print(
    "Training edges:",
    len(train_indices)
)

print(
    "Testing edges:",
    len(test_indices)
)


# ==================================================
# 7. CLASS DISTRIBUTION
# ==================================================

train_labels = (
    graph.y[train_indices]
)

normal_count = (
    train_labels == 0
).sum().item()

fraud_count = (
    train_labels == 1
).sum().item()


print("\nTraining class distribution:")

print(
    "Normal:",
    normal_count
)

print(
    "Fraud:",
    fraud_count
)


# ==================================================
# 8. LOSS
# ==================================================

# Moderate fraud weighting

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


# ==================================================
# 9. OPTIMIZER
# ==================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ==================================================
# 10. TRAIN
# ==================================================

epochs = 25

best_f1 = 0.0


print("\n======================================")
print("STARTING IMPROVED GNN TRAINING")
print("======================================\n")


for epoch in range(
    1,
    epochs + 1
):

    model.train()

    optimizer.zero_grad()


    # Forward

    output = model(
        graph.x,
        graph.edge_index,
        graph.edge_attr
    )


    # Loss

    loss = criterion(
        output[train_indices],
        graph.y[train_indices]
    )


    # Backpropagation

    loss.backward()


    # Gradient clipping

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )


    optimizer.step()


    # ==================================================
    # EVALUATION
    # ==================================================

    model.eval()

    with torch.no_grad():

        test_output = model(
            graph.x,
            graph.edge_index,
            graph.edge_attr
        )

        probabilities = torch.softmax(
            test_output[test_indices],
            dim=1
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()


    y_true = (
        graph.y[test_indices]
        .detach()
        .cpu()
        .numpy()
    )

    y_pred = (
        predictions
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

    roc_auc = roc_auc_score(
        y_true,
        y_probability
    )

    pr_auc = average_precision_score(
        y_true,
        y_probability
    )


    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {loss.item():.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"ROC-AUC: {roc_auc:.4f} | "
        f"PR-AUC: {pr_auc:.4f}"
    )


    # Save best model

    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            model.state_dict(),
            "../models/improved_gnn_model.pth"
        )


# ==================================================
# 11. SAVE FINAL MODEL
# ==================================================

torch.save(
    model.state_dict(),
    "../models/improved_gnn_final.pth"
)


print("\n======================================")
print("IMPROVED GNN TRAINING COMPLETED")
print("======================================")

print(
    "Best F1:",
    round(best_f1, 4)
)

print(
    "\nBest model:"
)

print(
    "models/improved_gnn_model.pth"
)

print(
    "\nFinal model:"
)

print(
    "models/improved_gnn_final.pth"
)