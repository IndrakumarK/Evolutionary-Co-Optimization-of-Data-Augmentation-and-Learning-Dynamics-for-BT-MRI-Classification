import copy
import torch
import numpy as np
import time
import csv
from tqdm import tqdm
from optimizer.fitness import compute_fitness


def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    augmentation_pipeline,
                    device):

    model.train()
    epoch_losses = []

    for images, targets in tqdm(dataloader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        images_aug = augmentation_pipeline(images)

        optimizer.zero_grad()
        outputs = model(images_aug)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses), epoch_losses


def evaluate_model(model,
                   dataloader,
                   augmentation_pipeline,
                   device,
                   task="classification"):

    model.eval()
    total_fitness = 0

    with torch.no_grad():
        for images, targets in dataloader:

            images = images.to(device)
            targets = targets.to(device)

            # Clean validation prediction
            outputs = model(images)

            # Fidelity comparison
            images_aug = augmentation_pipeline(images)

            fitness = compute_fitness(
                images,
                images_aug,
                outputs,
                targets,
                task=task
            )

            total_fitness += fitness.item()

    return total_fitness / len(dataloader)


def evolutionary_loop(model,
                      train_loader,
                      val_loader,
                      etdacvo,
                      augmentation_pipeline,
                      optimizer_class,
                      device="cuda",
                      generations=30,
                      task="classification",
                      log_csv_path="experiments/runtime_log.csv"):

    base_model = model
    convergence_records = []

    # CSV header
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Runtime_sec"])

    total_start = time.time()

    for gen in range(generations):

        gen_start = time.time()
        print(f"\n===== Generation {gen+1}/{generations} =====")

        fitness_scores = []
        best_score = float("inf")
        best_model_state = None

        for idx, theta in enumerate(etdacvo.population):

            model_i = copy.deepcopy(base_model).to(device)

            params = etdacvo.decode_theta(theta)
            augmentation_pipeline.update_params(params)

            optimizer_i = optimizer_class(
                model_i.parameters(),
                lr=params["lr"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"]
            )

            train_loss, loss_history = train_one_epoch(
                model_i,
                train_loader,
                optimizer_i,
                augmentation_pipeline,
                device
            )

            val_fitness = evaluate_model(
                model_i,
                val_loader,
                augmentation_pipeline,
                device,
                task=task
            )

            fitness_scores.append(val_fitness)
            convergence_records.extend(loss_history)

            print(f"Individual {idx} | Fitness={val_fitness:.4f}")

            if val_fitness < best_score:
                best_score = val_fitness
                best_model_state = copy.deepcopy(model_i.state_dict())

            del model_i
            del optimizer_i
            torch.cuda.empty_cache()

        etdacvo.update_population(fitness_scores)
        base_model.load_state_dict(best_model_state)

        gen_end = time.time()
        gen_runtime = gen_end - gen_start

        print(f"Generation {gen+1} Runtime: {gen_runtime:.2f} sec")

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen+1, gen_runtime])

    total_end = time.time()
    total_runtime = total_end - total_start

    print(f"\nTotal Evolution Runtime: {total_runtime:.2f} sec")

    best_theta = etdacvo.get_best()
    best_params = etdacvo.decode_theta(best_theta)

    return best_params, convergence_records, total_runtime