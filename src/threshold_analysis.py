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

print("Loading graph...")

graph = torch.load(
    "../data/transaction_graph.pt",
    weights_only=False
)

print("Graph loaded!")

print("Nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)


# ==================================================
# 2. DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

graph = graph.to(device)

print("Device:", device)


# ==================================================
# 3. NORMALIZE NODE FEATURES
# ==================================================

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


# ==================================================
# 4. NORMALIZE EDGE FEATURES
# ==================================================

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
# 5. SAME MODEL ARCHITECTURE
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

        source = edge_index[0]

        destination = edge_index[1]

        source_embedding = x[source]

        destination_embedding = x[destination]

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
# 6. CREATE MODEL
# ==================================================

model = GraphSAGE(
    node_features=graph.x.shape[1],
    edge_features=graph.edge_attr.shape[1],
    hidden_channels=64
)

model = model.to(device)


# ==================================================
# 7. LOAD BEST MODEL
# ==================================================

model.load_state_dict(
    torch.load(
        "../models/improved_gnn_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print(
    "Best improved model loaded!"
)


# ==================================================
# 8. RECREATE TEST SPLIT
# ==================================================

torch.manual_seed(42)

indices = torch.randperm(
    graph.num_edges,
    device=device
)

train_size = int(
    0.8 * graph.num_edges
)

test_indices = indices[
    train_size:
]


# ==================================================
# 9. GET FRAUD PROBABILITIES
# ==================================================

print("\nGenerating fraud probabilities...")

with torch.no_grad():

    output = model(
        graph.x,
        graph.edge_index,
        graph.edge_attr
    )

    probabilities = torch.softmax(
        output[test_indices],
        dim=1
    )[:, 1]


# ==================================================
# 10. TRUE LABELS
# ==================================================

y_true = (
    graph.y[test_indices]
    .cpu()
    .numpy()
)

probabilities = (
    probabilities
    .cpu()
    .numpy()
)


# ==================================================
# 11. RANKING METRICS
# ==================================================

roc_auc = roc_auc_score(
    y_true,
    probabilities
)

pr_auc = average_precision_score(
    y_true,
    probabilities
)

print("\n======================================")
print("RANKING PERFORMANCE")
print("======================================")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)


# ==================================================
# 12. THRESHOLD ANALYSIS
# ==================================================

thresholds = [
    0.50,
    0.30,
    0.20,
    0.10,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001
]


print("\n======================================")
print("THRESHOLD ANALYSIS")
print("======================================")

print(
    "\nThreshold | Precision | Recall | F1 | Alerts"
)

print(
    "-" * 55
)


best_threshold = None
best_f1 = 0.0


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    alerts = predictions.sum()


    print(
        f"{threshold:9.3f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f} | "
        f"{alerts:6d}"
    )


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold


# ==================================================
# 13. BEST THRESHOLD
# ==================================================

print("\n======================================")
print("BEST THRESHOLD")
print("======================================")

print(
    "Threshold:",
    best_threshold
)

print(
    "F1 Score:",
    round(best_f1, 4)
)