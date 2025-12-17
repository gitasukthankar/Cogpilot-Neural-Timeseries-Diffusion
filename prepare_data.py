import os
from pathlib import Path
from CogPilot.preprocessing import generate_expert_labels, index_dataset
import logging
log = logging.getLogger(__name__)

# CONFIGURATION 
RAW_DATA_ROOT = "dataPackage/task-ils"  

# Output paths (Must match what is in your conf/dataset/cogpilot.yaml)
PROCESSED_DIR = Path("data/processed")
LABELS_PATH = PROCESSED_DIR / "subject_labels.json"
INDEX_PATH = PROCESSED_DIR / "dataset_index.json"

def main():
    #  Create output directory
    if not PROCESSED_DIR.exists():
        print(f"Creating directory: {PROCESSED_DIR}")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    #  Generate Expert/Novice Labels
    print(f"Generating labels from {RAW_DATA_ROOT}...")
    try:
        generate_expert_labels(root_dir=RAW_DATA_ROOT, output_path=str(LABELS_PATH))
        print(f"Saved labels to {LABELS_PATH}")
    except Exception as e:
        print(f"Error generating labels: {e}")
        return

    # Index the Dataset
    print(f"Indexing dataset...")
    try:
        index_dataset(
            data_root=RAW_DATA_ROOT, 
            labels_path=str(LABELS_PATH), 
            output_json=str(INDEX_PATH)
        )
        print(f"Saved index to {INDEX_PATH}")
    except Exception as e:
        print(f"Error indexing dataset: {e}")
        return

    print("Data preparation complete. Ready for training.")

if __name__ == "__main__":
    main()