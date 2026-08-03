import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from neuracar.train import build_dataset, evaluate, train_reuse_net


def main():
    parser = argparse.ArgumentParser(description="Train the ReuseNet feedforward NN on a trace.")
    parser.add_argument("trace_path", help="Path to a time,obj_id,size CSV trace")
    parser.add_argument("--window", type=float, required=True, help="Reuse window (same time units as trace)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = build_dataset(args.trace_path, args.window, args.train_frac)
    model, history = train_reuse_net(X_train, y_train, X_test, y_test, epochs=args.epochs, lr=args.lr)

    metrics = evaluate(model, X_test, y_test)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final test loss:  {history['test_loss'][-1]:.4f}")
    print(f"Test accuracy:    {metrics['accuracy']:.4f}")
    print(f"Test AUC:         {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
