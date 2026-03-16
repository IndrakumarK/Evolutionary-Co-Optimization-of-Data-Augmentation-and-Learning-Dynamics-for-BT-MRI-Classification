import json
from training.evolutionary_loop import evolutionary_loop
from analysis.convergence_analysis import (
    convergence_epoch,
    oscillation_amplitude,
    curvature_metric
)


def train(model,
          train_loader,
          val_loader,
          etdacvo,
          augmentation_pipeline,
          optimizer_class,
          device="cuda",
          save_path="experiments/best_params.json"):

    best_params, convergence_history, total_runtime = evolutionary_loop(
        model,
        train_loader,
        val_loader,
        etdacvo,
        augmentation_pipeline,
        optimizer_class,
        device=device
    )

    model.eval()

    print("\n===== Convergence Analysis =====")
    print("Convergence Epoch:",
          convergence_epoch(convergence_history))
    print("Oscillation Amplitude:",
          oscillation_amplitude(convergence_history))
    print("Curvature Metric:",
          curvature_metric(convergence_history))
    print("Total Evolution Runtime:", total_runtime)

    with open(save_path, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"Best parameters saved to {save_path}")

    return best_params