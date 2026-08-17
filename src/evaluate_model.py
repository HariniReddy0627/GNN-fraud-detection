import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


print("Loading graph...")

graph = torch.load(
    "../data/transaction_graph.pt",
    weights_only=False
)

print("Graph loaded successfully!")

print("Nodes:", graph.num_nodes)
print("Edges:", graph.num_edges)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

graph = graph.to(device)

print("Device:", device)


# Normalize edge features

edge_attr = graph.edge_attr.clone()

amount = edge_attr[:, 0]
step = edge_attr[:, 1]

amount = torch.log1p(amount)

amount = (
    amount - amount.mean()
) / (
    amount.std() + 1e-8
)

step = (
    step - step.mean()
) / (
    step.std() + 1e-8
)

graph.edge_attr = torch.stack(
    [amount, step],
    dim=1
)


# GraphSAGE model

class GraphSAGE(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels
    ):

        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_channels * 2 + 2,
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


# Create model

model = GraphSAGE(
    in_channels=graph.x.shape[1],
    hidden_channels=32
)

model = model.to(device)

print("Model created successfully!")


# Load trained model

model.load_state_dict(
    torch.load(
        "../models/best_gnn_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Best GNN model loaded successfully!")


# Same test split

torch.manual_seed(42)

indices = torch.randperm(
    graph.num_edges,
    device=device
)

train_size = int(
    0.8 * graph.num_edges
)

test_indices = indices[train_size:]

print("Test edges:", len(test_indices))


# Predictions

print("Running predictions...")

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

    predictions = (
        probabilities >= 0.5
    ).long()


# Convert to CPU

y_true = (
    graph.y[test_indices]
    .cpu()
    .numpy()
)

y_pred = (
    predictions
    .cpu()
    .numpy()
)

y_probability = (
    probabilities
    .cpu()
    .numpy()
)


# Metrics

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


# Results

print("\n======================================")
print("FRAUD DETECTION RESULTS")
print("======================================")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# Confusion matrix

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)


# Classification report

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Fraud"
        ],
        zero_division=0
    )
)