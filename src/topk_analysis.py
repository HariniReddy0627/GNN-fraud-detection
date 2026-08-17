import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from sklearn.metrics import roc_auc_score, average_precision_score


# ==================================================
# 1. LOAD GRAPH
# ==================================================

print("Loading graph...")

graph = torch.load(
    "../data/transaction_graph.pt",
    weights_only=False
)

print("Graph loaded successfully!")

print("Nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)


# ==================================================
# 2. DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
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
# 5. MODEL
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
# 6. LOAD BEST MODEL
# ==================================================

model = GraphSAGE(
    node_features=graph.x.shape[1],
    edge_features=graph.edge_attr.shape[1],
    hidden_channels=64
)

model = model.to(device)

model.load_state_dict(
    torch.load(
        "../models/improved_gnn_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Best model loaded!")


# ==================================================
# 7. SAME TEST SPLIT
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
# 8. GENERATE SCORES
# ==================================================

print("Generating risk scores...")

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
# 9. CPU / NUMPY
# ==================================================

scores = (
    probabilities
    .cpu()
    .numpy()
)

labels = (
    graph.y[test_indices]
    .cpu()
    .numpy()
)


# ==================================================
# 10. OVERALL METRICS
# ==================================================

roc_auc = roc_auc_score(
    labels,
    scores
)

pr_auc = average_precision_score(
    labels,
    scores
)

print("\n======================================")
print("OVERALL MODEL RANKING")
print("======================================")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)


# ==================================================
# 11. TOP-K ANALYSIS
# ==================================================

ranking = scores.argsort()[::-1]

top_k_values = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000
]


print("\n======================================")
print("TOP-K FRAUD ANALYSIS")
print("======================================")

print(
    "\nTop-K | Fraud Found | Precision | Recall"
)

print(
    "-" * 50
)


total_fraud = labels.sum()


for k in top_k_values:

    k = min(
        k,
        len(labels)
    )

    top_indices = ranking[:k]

    fraud_found = labels[
        top_indices
    ].sum()

    precision = (
        fraud_found / k
    )

    recall = (
        fraud_found / total_fraud
        if total_fraud > 0
        else 0
    )

    print(
        f"{k:5d} | "
        f"{fraud_found:11d} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f}"
    )


# ==================================================
# 12. TOP TRANSACTIONS
# ==================================================

print("\n======================================")
print("TOP 10 HIGHEST-RISK TRANSACTIONS")
print("======================================")

top10 = ranking[:10]

for position, index in enumerate(
    top10,
    start=1
):

    print(
        f"Rank {position:02d} | "
        f"Score: {scores[index]:.6f} | "
        f"Actual Fraud: {labels[index]}"
    )