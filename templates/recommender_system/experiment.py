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
# RECOMMENDATION INTERACTION MATRIX & DATASET GENERATOR
# =========================================================================
def generate_recommendation_data(num_users=300, num_items=500, num_interactions=4000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Power-law popularity distribution
    user_ids = np.random.randint(0, num_users, size=num_interactions)
    item_probs = 1.0 / (np.arange(1, num_items + 1) ** 0.8)
    item_probs /= item_probs.sum()
    item_ids = np.random.choice(num_items, size=num_interactions, p=item_probs)

    user_item_set = set(zip(user_ids, item_ids))
    interactions = list(user_item_set)

    # Train / Val / Test split
    np.random.shuffle(interactions)
    n = len(interactions)
    train_data = interactions[: int(0.7 * n)]
    val_data = interactions[int(0.7 * n) : int(0.85 * n)]
    test_data = interactions[int(0.85 * n) :]

    return num_users, num_items, train_data, val_data, test_data


# =========================================================================
# NEURAL COLLABORATIVE FILTERING / LIGHTGCN BASELINE
# =========================================================================
class NeuralRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NeuralRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, users, items):
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        x = torch.cat([u_emb, i_emb], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score


# =========================================================================
# METRICS: NDCG@K and Recall@K
# =========================================================================
def evaluate_ranking(model, eval_data, num_items, device, k=10):
    model.eval()
    user_to_items = {}
    for u, i in eval_data:
        user_to_items.setdefault(u, set()).add(i)

    ndcg_list = []
    recall_list = []

    with torch.no_grad():
        for u, true_items in list(user_to_items.items())[:50]:  # Eval subset
            user_tensor = torch.tensor([u] * num_items, device=device)
            item_tensor = torch.arange(num_items, device=device)
            scores = model(user_tensor, item_tensor).cpu().numpy()
            top_k_items = np.argsort(scores)[-k:][::-1]

            hits = len(set(top_k_items).intersection(true_items))
            recall = hits / max(len(true_items), 1)
            recall_list.append(recall)

            dcg = 0.0
            for rank, item in enumerate(top_k_items):
                if item in true_items:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(true_items))))
            ndcg = dcg / max(idcg, 1e-9)
            ndcg_list.append(ndcg)

    return float(np.mean(ndcg_list)), float(np.mean(recall_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--embedding_dim", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_users, num_items, train_data, val_data, test_data = generate_recommendation_data()
    model = NeuralRecommender(num_users, num_items, embedding_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_users = torch.tensor([u for u, _ in train_data], device=device)
    train_items = torch.tensor([i for _, i in train_data], device=device)

    history = {"train_loss": [], "val_ndcg": [], "val_recall": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        # Positive interactions
        pos_scores = model(train_users, train_items)
        # Negative sampling
        neg_items = torch.randint(0, num_items, (len(train_users),), device=device)
        neg_scores = model(train_users, neg_items)

        loss = -torch.mean(torch.log(pos_scores + 1e-9) + torch.log(1 - neg_scores + 1e-9))
        loss.backward()
        optimizer.step()

        val_ndcg, val_recall = evaluate_ranking(model, val_data, num_items, device, k=10)
        history["train_loss"].append(float(loss.item()))
        history["val_ndcg"].append(val_ndcg)
        history["val_recall"].append(val_recall)

    test_ndcg, test_recall = evaluate_ranking(model, test_data, num_items, device, k=10)
    elapsed = time.time() - start_time

    final_info = {
        "final_test_ndcg@10": test_ndcg,
        "final_test_recall@10": test_recall,
        "final_train_loss": history["train_loss"][-1],
        "training_time_sec": elapsed,
        "epochs": args.epochs,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w", encoding="utf-8") as f:
        json.dump(final_info, f, indent=2)

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[RECSYS COMPLETED] Test NDCG@10: {test_ndcg*100:.2f}%, Recall@10: {test_recall*100:.2f}% ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
