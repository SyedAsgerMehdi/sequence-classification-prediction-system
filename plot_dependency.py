from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("outputs")
SEQUENCE_LENGTHS = (10, 60, 80)


def main():
    frames = []
    for sequence_length in SEQUENCE_LENGTHS:
        metrics = pd.read_csv(
            OUTPUT_DIR / (
                "metrics.csv"
                if sequence_length == 60
                else f"metrics_seq{sequence_length}.csv"
            )
        )
        metrics.insert(0, "Sequence Length", sequence_length)
        frames.append(metrics)

    dependency_metrics = pd.concat(frames, ignore_index=True)
    dependency_metrics.to_csv(OUTPUT_DIR / "dependency_metrics.csv", index=False)

    plt.figure(figsize=(7, 4))
    for model, group in dependency_metrics.groupby("Model"):
        ordered = group.sort_values("Sequence Length")
        plt.plot(
            ordered["Sequence Length"],
            ordered["Validation Accuracy"],
            marker="o",
            label=model,
        )
    plt.xlabel("Sequence Length")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Sequence Length")
    plt.xticks(SEQUENCE_LENGTHS)
    plt.legend(title="Architecture")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dependency_accuracy.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()