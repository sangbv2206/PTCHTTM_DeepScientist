import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =========================================================================
# GRAPH DATASET GENERATOR (Benchmark Graph with Topological Structure)
# =========================================================================
def generate_synthetic_graph(num_nodes=600, num_features=64, num_classes=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Node features
    X = torch.randn(num_nodes, num_features)
    # Ground truth labels
    labels = torch.randint(0, num_classes, (num_nodes,))

    # Cluster-based adjacency matrix (Homophily graph)
    adj = torch.zeros(num_nodes, num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Higher probability if same class
            p = 0.08 if labels[i] == labels[j] else 0.005
            if np.random.rand() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Add self-loops
    adj += torch.eye(num_nodes)

    # Degree normalization: D^(-0.5) * A * D^(-0.5)
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_mat = torch.diag(deg_inv_sqrt)
    norm_adj = torch.mm(torch.mm(D_mat, adj), D_mat)

    # Train / Val / Test masks
    indices = torch.randperm(num_nodes)
    train_mask = indices[: int(0.6 * num_nodes)]
    val_mask = indices[int(0.6 * num_nodes) : int(0.8 * num_nodes)]
    test_mask = indices[int(0.8 * num_nodes) :]

    return X, norm_adj, labels, train_mask, val_mask, test_mask


# =========================================================================
# MODEL DEFINITION (GNN Layer & Network)
# =========================================================================
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support) + self.bias
        return output


class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, dropout=0.5):
        super(GNNModel, self).__init__()
        self.gc1 = GraphConvolution(in_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        out = self.gc2(h, adj)
        return out, h  # Return logits and embeddings for visualization


# =========================================================================
# MAIN TRAINING LOOP
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, adj, labels, train_idx, val_idx, test_idx = generate_synthetic_graph()
    X, adj, labels = X.to(device), adj.to(device), labels.to(device)

    model = GNNModel(X.shape[1], args.hidden_dim, int(labels.max().item()) + 1, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "test_acc": []}

    start_time = time.time()
    best_val_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X, adj)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(X, adj)
            val_loss = F.cross_entropy(eval_logits[val_idx], labels[val_idx]).item()
            val_preds = eval_logits[val_idx].argmax(dim=1)
            val_acc = (val_preds == labels[val_idx]).float().mean().item()

            test_preds = eval_logits[test_idx].argmax(dim=1)
            test_acc = (test_preds == labels[test_idx]).float().mean().item()

        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["test_acc"].append(float(test_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc

    elapsed = time.time() - start_time

    # Save final info and history
    final_info = {
        "final_test_accuracy": final_test_acc,
        "best_val_accuracy": best_val_acc,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "training_time_sec": elapsed,
        "epochs": args.epochs,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w", encoding="utf-8") as f:
        json.dump(final_info, f, indent=2)

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[EXPERIMENT COMPLETED] Best Val Acc: {best_val_acc*100:.2f}%, Test Acc: {final_test_acc*100:.2f}% ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
