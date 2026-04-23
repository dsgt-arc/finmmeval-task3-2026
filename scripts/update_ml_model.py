from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_making.ml_model.ml_model_manager import get_model_manager

"""Daily ML model update script for online learning."""


def main():
    """Update ML model if retraining needed."""
    print("Checking for model updates...")

    manager = get_model_manager()
    should_retrain, retrain_type = manager.should_retrain()

    if should_retrain:
        print(f"Performing {retrain_type} retrain...")
        manager.warm_start_retrain()
        manager.save()
        print(f"\u2713 Model updated ({retrain_type})")
    else:
        complete = len([o for o in manager.observations if o["target"] is not None])
        print(f"No update needed ({complete} observations)")


if __name__ == "__main__":
    main()
