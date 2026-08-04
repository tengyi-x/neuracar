import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuracar.model import FEATURES
from neuracar.train import build_dataset, evaluate, train_reuse_net


def main():
    parser = argparse.ArgumentParser(description="Feature ablation for ReuseNet: drop each feature one at a time.")
    parser.add_argument("trace_path")
    parser.add_argument("--window", type=float, required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    feature_sets = {"all": FEATURES}
    for name in FEATURES:
        remaining = [f for f in FEATURES if f != name]
        feature_sets[f"without_{name}"] = remaining

    print(f"{'feature set':<22} {'accuracy':>10} {'auc':>10}")
    for label, mask in feature_sets.items():
        X_train, y_train, X_test, y_test = build_dataset(
            args.trace_path, args.window, args.train_frac, feature_mask=mask
        )
        model, _ = train_reuse_net(X_train, y_train, X_test, y_test, epochs=args.epochs)
        metrics = evaluate(model, X_test, y_test)
        print(f"{label:<22} {metrics['accuracy']:>10.4f} {metrics['auc']:>10.4f}")


if __name__ == "__main__":
    main()
