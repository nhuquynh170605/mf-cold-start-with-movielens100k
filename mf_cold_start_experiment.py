
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# MF BASELINE - MovieLens 100K
#
# Experiments:
#   1) Random 80/10/10
#   2) Temporal 80/10/10
#
# Cold-start:
#   k=10,5,2,1  -> users with train_interactions < k
#   k=0         -> special case: users with 0 train interactions
#
# Outputs:
#   results/mf_cold_start/
#       summary.csv
#       cold_start_random.csv
#       cold_start_temporal.csv
#       training_random.csv
#       training_temporal.csv
#       learning_curve_random.png
#       learning_curve_temporal.png
#       split_statistics.csv
# ============================================================


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_movielens(data_dir):
    path = Path(data_dir) / "ml-100k" / "u.data"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Expected MovieLens 100K at "
            f"data/ml-100k/u.data"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    if df.isna().any().any():
        raise ValueError(f"MovieLens data contains missing values: {path}")
    if not df["rating"].between(1, 5).all():
        raise ValueError("MovieLens ratings must be in the range [1, 5]")
    if df.duplicated().any():
        raise ValueError("MovieLens data contains duplicate interactions")

    return df


def split_random(df, seed=42):
    """Interaction-level random 80/10/10 split."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(shuffled)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train = shuffled.iloc[:n_train].copy()
    val = shuffled.iloc[n_train:n_train + n_val].copy()
    test = shuffled.iloc[n_train + n_val:].copy()

    return train, val, test


def split_temporal(df):
    """Chronological 80/10/10 split. No shuffle."""
    ordered = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    n = len(ordered)
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)

    train = ordered.iloc[:n_train].copy()
    val = ordered.iloc[n_train:n_train + n_val].copy()
    test = ordered.iloc[n_train + n_val:].copy()

    return train, val, test


def encode_ids(train, val, test):
    """
    Encode users/items using IDs seen anywhere in the full dataset.
    The model itself is still trained only on train interactions.

    This makes it possible to identify unknown-to-train users/items
    during validation/test without crashing the embedding lookup.
    """
    all_users = sorted(
        set(train.user_id) | set(val.user_id) | set(test.user_id)
    )
    all_items = sorted(
        set(train.item_id) | set(val.item_id) | set(test.item_id)
    )

    user_map = {u: i for i, u in enumerate(all_users)}
    item_map = {m: i for i, m in enumerate(all_items)}

    def convert(df):
        out = df.copy()
        out["u"] = out["user_id"].map(user_map).astype(int)
        out["i"] = out["item_id"].map(item_map).astype(int)
        return out

    return convert(train), convert(val), convert(test), user_map, item_map


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)

        dot = (u * i).sum(dim=1)
        ub = self.user_bias(users).squeeze(1)
        ib = self.item_bias(items).squeeze(1)

        return self.global_bias + dot + ub + ib


def make_loader(df, batch_size=256, shuffle=True):
    users = torch.tensor(df["u"].values, dtype=torch.long)
    items = torch.tensor(df["i"].values, dtype=torch.long)
    ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    ds = TensorDataset(users, items, ratings)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predict_known(model, df, device):
    if len(df) == 0:
        return np.array([])

    users = torch.tensor(df["u"].values, dtype=torch.long, device=device)
    items = torch.tensor(df["i"].values, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        pred = model(users, items).detach().cpu().numpy()

    return pred


def rmse_mae(model, df, device):
    if len(df) == 0:
        return np.nan, np.nan

    pred = predict_known(model, df, device)
    y = df["rating"].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    return rmse, mae


def metric_values(predictions, ratings):
    if len(ratings) == 0:
        return np.nan, np.nan

    errors = predictions - ratings
    return (
        float(np.sqrt(np.mean(errors ** 2))),
        float(np.mean(np.abs(errors))),
    )


def fit_mean_baselines(train):
    """Fit global, user-mean, and item-mean rating baselines on train only."""
    global_mean = float(train["rating"].mean())
    user_means = train.groupby("user_id")["rating"].mean().to_dict()
    item_means = train.groupby("item_id")["rating"].mean().to_dict()
    return global_mean, user_means, item_means


def predict_baseline(df, baseline_name, global_mean, user_means, item_means):
    if baseline_name == "global_mean":
        return np.full(len(df), global_mean, dtype=float)

    if baseline_name == "user_mean":
        return df["user_id"].map(user_means).fillna(global_mean).to_numpy(
            dtype=float
        )

    if baseline_name == "item_mean":
        return df["item_id"].map(item_means).fillna(global_mean).to_numpy(
            dtype=float
        )

    raise ValueError(f"Unknown baseline: {baseline_name}")


def predict_bias_fallback(df, global_mean, user_means, item_means):
    """Use learned user/item means, falling back to the available side."""
    user_prediction = df["user_id"].map(user_means)
    item_prediction = df["item_id"].map(item_means)
    prediction = user_prediction.combine_first(item_prediction)
    return prediction.fillna(global_mean).to_numpy(dtype=float)


def baseline_metrics(df, predictions, name, split_name):
    rmse, mae = metric_values(
        predictions,
        df["rating"].to_numpy(dtype=float),
    )
    return {
        "split": split_name,
        "model": name,
        "n_interactions": len(df),
        "rmse": rmse,
        "mae": mae,
    }


def fallback_coverage_metrics(
    df,
    predictions,
    train_user_ids,
    train_item_ids,
    model_name,
    split_name,
):
    """Evaluate a total-coverage predictor separately on known/unseen groups."""
    user_known = df["user_id"].isin(train_user_ids)
    item_known = df["item_id"].isin(train_item_ids)
    groups = np.select(
        [user_known & item_known, ~user_known & item_known, user_known & ~item_known],
        ["known_user_item", "unseen_user", "unseen_item"],
        default="unseen_user_item",
    )

    work = df.copy()
    work["coverage_group"] = groups
    work["prediction"] = predictions
    rows = []
    for group, group_df in work.groupby("coverage_group", sort=False):
        rmse, mae = metric_values(
            group_df["prediction"].to_numpy(dtype=float),
            group_df["rating"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "split": split_name,
                "model": model_name,
                "coverage_group": group,
                "n_interactions": len(group_df),
                "rmse": rmse,
                "mae": mae,
            }
        )
    return pd.DataFrame(rows)


def get_train_user_counts(train):
    return train.groupby("user_id").size().to_dict()


def cold_start_masks(df, train_user_counts):
    counts = df["user_id"].map(train_user_counts).fillna(0).astype(int)
    return counts


def cold_start_table(model, df, train_user_counts, device, split_name):
    """
    Evaluate RMSE/MAE by number of interactions available for the user
    in TRAIN.

    Groups:
      unseen: 0
      1
      2-4
      5-9
      10+
    """
    if len(df) == 0:
        return pd.DataFrame()

    work = df.copy()
    work["train_interactions"] = cold_start_masks(
        work, train_user_counts
    )

    known = work["train_interactions"] > 0

    rows = []

    # Explicit cold-start bins.
    bins = [
        ("0_unseen", 0, 0),
        ("1", 1, 1),
        ("2-4", 2, 4),
        ("5-9", 5, 9),
        ("10+", 10, 10**9),
    ]

    for label, low, high in bins:
        g = work[
            (work["train_interactions"] >= low)
            & (work["train_interactions"] <= high)
        ].copy()

        if len(g) == 0:
            continue

        # MF has no learned user embedding for users with zero train data.
        if low == 0:
            rmse = np.nan
            mae = np.nan
            evaluable = 0
        else:
            rmse, mae = rmse_mae(model, g, device)
            evaluable = len(g)

        rows.append(
            {
                "split": split_name,
                "group": label,
                "n_interactions": len(g),
                "n_users": g["user_id"].nunique(),
                "evaluable_by_mf": evaluable,
                "rmse": rmse,
                "mae": mae,
                "mean_train_interactions":
                    float(g["train_interactions"].mean()),
            }
        )

    return pd.DataFrame(rows)


def cold_k_table(model, df, train_user_counts, device, split_name):
    """
    For k = 10,5,2,1:
      cold-start = train interaction count < k

    For k = 0:
      special thesis convention:
      cold-start = train interaction count == 0

    This special handling is necessary because count < 0 is impossible.
    """
    work = df.copy()
    work["train_interactions"] = cold_start_masks(
        work, train_user_counts
    )

    rows = []

    for k in [10, 5, 2, 1]:
        g = work[work["train_interactions"] < k]

        n_users = g["user_id"].nunique()
        n_rows = len(g)

        # Some users may have 0 interactions in train.
        # MF cannot directly evaluate those users.
        known = g[g["train_interactions"] > 0]

        if len(known) > 0:
            rmse, mae = rmse_mae(model, known, device)
        else:
            rmse, mae = np.nan, np.nan

        rows.append(
            {
                "split": split_name,
                "k": k,
                "definition": f"train_interactions < {k}",
                "n_test_interactions": n_rows,
                "n_users": n_users,
                "n_users_unseen": int(
                    (g["train_interactions"] == 0).sum()
                    and g.loc[
                        g["train_interactions"] == 0,
                        "user_id",
                    ].nunique()
                ),
                "rmse_known_users": rmse,
                "mae_known_users": mae,
            }
        )

    # k=0: special "unseen user" experiment.
    g = work[work["train_interactions"] == 0]

    rows.append(
        {
            "split": split_name,
            "k": 0,
            "definition": "train_interactions == 0 (unseen user)",
            "n_test_interactions": len(g),
            "n_users": g["user_id"].nunique(),
            "n_users_unseen": g["user_id"].nunique(),
            "rmse_known_users": np.nan,
            "mae_known_users": np.nan,
        }
    )

    return pd.DataFrame(rows)


def train_mf(
    train,
    val,
    n_users,
    n_items,
    epochs=20,
    batch_size=256,
    dim=64,
    lr=1e-3,
    seed=42,
    patience=3,
    checkpoint_path=None,
):
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        dim=dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = make_loader(
        train,
        batch_size=batch_size,
        shuffle=True,
    )

    history = []
    best_val_rmse = np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for users, items, ratings in loader:
            users = users.to(device)
            items = items.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            pred = model(users, items)
            loss = criterion(pred, ratings)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(ratings)
            total_n += len(ratings)

        train_loss = total_loss / total_n

        val_rmse, val_mae = rmse_mae(model, val, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_RMSE={val_rmse:.4f} | "
            f"val_MAE={val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_rmse": val_rmse,
                        "seed": seed,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best_epoch={best_epoch}, "
                    f"best_val_RMSE={best_val_rmse:.4f}"
                )
                break

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    history_df = pd.DataFrame(history)
    return model, history_df, device, best_epoch, best_val_rmse


def split_statistics(train, val, test, name):
    train_users = set(train.user_id)
    val_users = set(val.user_id)
    test_users = set(test.user_id)

    train_items = set(train.item_id)
    val_items = set(val.item_id)
    test_items = set(test.item_id)

    return {
        "split": name,
        "train_interactions": len(train),
        "val_interactions": len(val),
        "test_interactions": len(test),
        "train_users": len(train_users),
        "val_users": len(val_users),
        "test_users": len(test_users),
        "val_unseen_users": len(val_users - train_users),
        "test_unseen_users": len(test_users - train_users),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "test_items": len(test_items),
        "val_unseen_items": len(val_items - train_items),
        "test_unseen_items": len(test_items - train_items),
    }


def evaluation_coverage(
    model,
    df,
    train_user_ids,
    train_item_ids,
    device,
    split_name,
):
    """Report coverage and metrics without treating unseen embeddings as learned."""
    if len(df) == 0:
        return pd.DataFrame()

    work = df.copy()
    user_known = work["user_id"].isin(train_user_ids)
    item_known = work["item_id"].isin(train_item_ids)
    work["coverage_group"] = np.select(
        [
            user_known & item_known,
            ~user_known & item_known,
            user_known & ~item_known,
        ],
        [
            "known_user_item",
            "unseen_user",
            "unseen_item",
        ],
        default="unseen_user_item",
    )

    rows = []
    for group, group_df in work.groupby("coverage_group", sort=False):
        evaluable = group == "known_user_item"
        if evaluable:
            rmse, mae = rmse_mae(model, group_df, device)
        else:
            rmse, mae = np.nan, np.nan

        rows.append(
            {
                "split": split_name,
                "coverage_group": group,
                "n_interactions": len(group_df),
                "n_users": group_df["user_id"].nunique(),
                "n_items": group_df["item_id"].nunique(),
                "evaluable_by_trained_mf": int(evaluable),
                "rmse": rmse,
                "mae": mae,
            }
        )

    return pd.DataFrame(rows)


def run_experiment(name, train_raw, val_raw, test_raw, args, out_dir, seed):
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name.upper()}")
    print("=" * 80)

    train, val, test, user_map, item_map = encode_ids(
        train_raw, val_raw, test_raw
    )

    stats = split_statistics(train_raw, val_raw, test_raw, name)

    print("\nSplit statistics:")
    for key, value in stats.items():
        if key != "split":
            print(f"  {key}: {value}")

    train_user_counts = get_train_user_counts(train_raw)

    # Add users with zero train interactions explicitly.
    all_users = set(train_raw.user_id) | set(val_raw.user_id) | set(test_raw.user_id)
    for u in all_users:
        train_user_counts.setdefault(u, 0)

    train_user_ids = set(train_raw.user_id)
    train_item_ids = set(train_raw.item_id)
    val_known = val[
        val["user_id"].isin(train_user_ids)
        & val["item_id"].isin(train_item_ids)
    ].copy()

    checkpoint_path = out_dir / f"mf_{name}_seed{seed}_best.pt"
    model, history, device, best_epoch, best_val_rmse = train_mf(
        train=train,
        val=val_known,
        n_users=len(user_map),
        n_items=len(item_map),
        epochs=args.epochs,
        batch_size=args.batch_size,
        dim=args.dim,
        lr=args.lr,
        seed=seed,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )

    # Overall rating prediction.
    train_rmse, train_mae = rmse_mae(model, train, device)
    val_rmse, val_mae = rmse_mae(model, val_known, device)

    # Test may contain unseen users/items. Evaluate only rows for which
    # both user and item occurred in training.
    test_known = test[
        test["user_id"].isin(train_user_ids)
        & test["item_id"].isin(train_item_ids)
    ].copy()

    test_rmse, test_mae = rmse_mae(model, test_known, device)

    global_mean, user_means, item_means = fit_mean_baselines(train_raw)
    baseline_rows = []
    fallback_coverage_rows = []
    for baseline_name in ["global_mean", "user_mean", "item_mean"]:
        predictions = predict_baseline(
            test_raw,
            baseline_name,
            global_mean,
            user_means,
            item_means,
        )
        baseline_rows.append(
            baseline_metrics(test_raw, predictions, baseline_name, name)
        )

    fallback_predictions = predict_bias_fallback(
        test_raw,
        global_mean,
        user_means,
        item_means,
    )
    baseline_rows.append(
        baseline_metrics(
            test_raw,
            fallback_predictions,
            "bias_fallback",
            name,
        )
    )
    fallback_coverage_rows.append(
        fallback_coverage_metrics(
            test_raw,
            fallback_predictions,
            train_user_ids,
            train_item_ids,
            "bias_fallback",
            name,
        )
    )

    mf_fallback_predictions = fallback_predictions.copy()
    known_positions = (
        test_raw["user_id"].isin(train_user_ids).to_numpy()
        & test_raw["item_id"].isin(train_item_ids).to_numpy()
    )
    mf_fallback_predictions[known_positions] = predict_known(
        model,
        test.loc[known_positions],
        device,
    )
    baseline_rows.append(
        baseline_metrics(
            test_raw,
            mf_fallback_predictions,
            "mf_with_bias_fallback",
            name,
        )
    )
    fallback_coverage_rows.append(
        fallback_coverage_metrics(
            test_raw,
            mf_fallback_predictions,
            train_user_ids,
            train_item_ids,
            "mf_with_bias_fallback",
            name,
        )
    )

    val_coverage = evaluation_coverage(
        model,
        val,
        train_user_ids,
        train_item_ids,
        device,
        f"{name}_validation",
    )
    test_coverage = evaluation_coverage(
        model,
        test,
        train_user_ids,
        train_item_ids,
        device,
        f"{name}_test",
    )

    overall = pd.DataFrame(
        [
            {
                "split": name,
                "train_rmse": train_rmse,
                "train_mae": train_mae,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_rows": len(val_raw),
                "val_rows_evaluable_by_mf": len(val_known),
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
                "test_rmse_known_users_items": test_rmse,
                "test_mae_known_users_items": test_mae,
                "test_rows": len(test_raw),
                "test_rows_evaluable_by_mf": len(test_known),
                "test_rows_unseen_user": int(
                    (~test_raw["user_id"].isin(train_user_ids)).sum()
                ),
                "test_rows_unseen_item": int(
                    (~test_raw["item_id"].isin(train_item_ids)).sum()
                ),
                "test_rows_unseen_user_item": int(
                    (
                        ~test_raw["user_id"].isin(train_user_ids)
                        & ~test_raw["item_id"].isin(train_item_ids)
                    ).sum()
                ),
            }
        ]
    )

    # Save training curve.
    history["split"] = name
    history["seed"] = seed
    history.to_csv(
        out_dir / f"training_{name}_seed{seed}.csv",
        sep="|",
        index=False,
    )

    cold_groups = cold_start_table(
        model,
        test,
        train_user_counts,
        device,
        name,
    )
    cold_groups.to_csv(
        out_dir / f"cold_start_groups_{name}.csv",
        sep="|",
        index=False,
    )

    cold_k = cold_k_table(
        model,
        test,
        train_user_counts,
        device,
        name,
    )
    cold_k.to_csv(
        out_dir / f"cold_start_k_{name}.csv",
        sep="|",
        index=False,
    )

    coverage = pd.concat([val_coverage, test_coverage], ignore_index=True)

    return (
        overall,
        history,
        stats,
        cold_groups,
        cold_k,
        coverage,
        pd.DataFrame(baseline_rows),
        pd.concat(fallback_coverage_rows, ignore_index=True),
    )


def plot_learning_curves(histories, out_dir):
    # Loss plot
    plt.figure(figsize=(9, 5))
    for name, h in histories.items():
        plt.plot(
            h["epoch"],
            h["train_loss"],
            marker="o",
            label=f"{name} train loss",
        )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Training Loss")
    plt.title("MF Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_train_loss.png", dpi=150)
    plt.close()

    # Validation RMSE
    plt.figure(figsize=(9, 5))
    for name, h in histories.items():
        plt.plot(
            h["epoch"],
            h["val_rmse"],
            marker="o",
            label=f"{name} validation RMSE",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation RMSE")
    plt.title("MF Validation RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_validation_rmse.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/mf_cold_start")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated seeds for multi-seed evaluation.",
    )
    parser.add_argument("--patience", type=int, default=3)

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.dim < 1:
        parser.error("--dim must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be greater than 0")
    if args.patience < 1:
        parser.error("--patience must be at least 1")

    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as error:
        parser.error(f"--seeds must be comma-separated integers: {error}")
    if not seeds:
        parser.error("--seeds must contain at least one seed")

    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_movielens(args.data_dir)

    print(f"Loaded MovieLens 100K: {len(df):,} interactions")
    print(f"Users: {df.user_id.nunique():,}")
    print(f"Items: {df.item_id.nunique():,}")

    results = []
    histories = {}
    all_stats = []
    all_cold_k = []
    all_coverage = []
    all_baselines = []
    all_fallback_coverage = []

    for seed in seeds:
        random_train, random_val, random_test = split_random(
            df,
            seed=seed,
        )
        temporal_train, temporal_val, temporal_test = split_temporal(df)

        for name, tr, va, te in [
            ("random", random_train, random_val, random_test),
            ("temporal", temporal_train, temporal_val, temporal_test),
        ]:
            (
                overall,
                history,
                stats,
                cold_groups,
                cold_k,
                coverage,
                baselines,
                fallback_coverage,
            ) = run_experiment(
                name,
                tr,
                va,
                te,
                args,
                out_dir,
                seed,
            )

            overall["seed"] = seed
            baselines["seed"] = seed
            fallback_coverage["seed"] = seed
            results.append(overall)
            all_baselines.append(baselines)
            all_fallback_coverage.append(fallback_coverage)
            histories[f"{name}_seed{seed}"] = history
            stats["seed"] = seed
            all_stats.append(stats)
            cold_k["seed"] = seed
            all_cold_k.append(cold_k)
            coverage["seed"] = seed
            all_coverage.append(coverage)

    summary = pd.concat(results, ignore_index=True)
    summary.to_csv(
        out_dir / "summary.csv",
        sep="|",
        index=False,
    )

    baseline_summary = pd.concat(all_baselines, ignore_index=True)
    baseline_summary.to_csv(
        out_dir / "baseline_results.csv",
        sep="|",
        index=False,
    )

    summary_mean_std = (
        summary.groupby("split")
        .agg(
            runs=("seed", "count"),
            test_rmse_mean=("test_rmse_known_users_items", "mean"),
            test_rmse_std=("test_rmse_known_users_items", "std"),
            test_mae_mean=("test_mae_known_users_items", "mean"),
            test_mae_std=("test_mae_known_users_items", "std"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reset_index()
    )
    summary_mean_std.to_csv(
        out_dir / "summary_mean_std.csv",
        sep="|",
        index=False,
    )

    baseline_mean_std = (
        baseline_summary.groupby(["split", "model"])
        .agg(
            runs=("seed", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
    )
    baseline_mean_std.to_csv(
        out_dir / "baseline_mean_std.csv",
        sep="|",
        index=False,
    )

    pd.concat(all_fallback_coverage, ignore_index=True).to_csv(
        out_dir / "fallback_coverage.csv",
        sep="|",
        index=False,
    )

    pd.DataFrame(all_stats).to_csv(
        out_dir / "split_statistics.csv",
        sep="|",
        index=False,
    )

    cold_k_all = pd.concat(all_cold_k, ignore_index=True)
    cold_k_all.to_csv(
        out_dir / "cold_start_k_all.csv",
        sep="|",
        index=False,
    )

    pd.concat(all_coverage, ignore_index=True).to_csv(
        out_dir / "evaluation_coverage.csv",
        sep="|",
        index=False,
    )

    plot_learning_curves(histories, out_dir)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nCold-start by k:")
    print(cold_k_all.to_string(index=False))

    print(f"\nResults saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
