import numpy as np
from utils.seed import set_seed
from analysis.variance_analysis import aggregate_seeds
from analysis.convergence_analysis import convergence_epoch
from training.trainer import train

SEEDS = [42, 52, 62, 72, 82]


def run_experiment(model_fn,
                   train_loader,
                   val_loader,
                   etdacvo,
                   augmentation_pipeline,
                   optimizer_class,
                   device):

    acc_results = []
    convergence_results = []

    for seed in SEEDS:

        print(f"\nRunning seed {seed}")
        set_seed(seed)

        model = model_fn().to(device)

        best_params = train(
            model,
            train_loader,
            val_loader,
            etdacvo,
            augmentation_pipeline,
            optimizer_class,
            device
        )

        acc = evaluate_accuracy(model, val_loader, device)
        acc_results.append(acc)

        convergence_results.append(best_params)

    mean_acc, std_acc = aggregate_seeds(acc_results)

    print("\n==== FINAL RESULTS ====")
    print("Mean Accuracy:", mean_acc)
    print("Std Accuracy:", std_acc)

    return mean_acc, std_acc