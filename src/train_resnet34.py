import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import create_model


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main(args):

    base_dir = Path(__file__).resolve().parent.parent

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, test_loader = create_dataloaders(
        base_dir,
        batch_size=args.batch_size
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    model = create_model(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_accuracy = 0.0

    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)

    checkpoint_path = (
        models_dir / "resnet34_best.pth"
    )

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Saved best model: "
                f"{checkpoint_path}"
            )

    print("\nTraining complete.")

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    # Load best model
    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"\nTest Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001
    )

    args = parser.parse_args()

    main(args)