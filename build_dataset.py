import argparse
import json
import random
from collections import defaultdict

from datasets import ClassLabel, Dataset
from tqdm import tqdm

from arxiv_taxonomy import label_id, label_maps, raw_records_labeled_pairs


def downsample(
    pairs: list[tuple[dict, str]],
    cap: int,
    seed: int,
) -> list[tuple[dict, str]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec, sup in pairs:
        by_class[sup].append(rec)
    out: list[tuple[dict, str]] = []
    for sup, recs in by_class.items():
        for r in rng.sample(recs, min(cap, len(recs))):
            out.append((r, sup))
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="raw_ds.json → data-dir (train.json, validation.json) + label maps",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    with open(f"{args.data_dir}/raw_ds.json", encoding="utf-8-sig") as f:
        records = json.load(f)
    pairs = raw_records_labeled_pairs(records)

    if args.max_per_class is not None:
        pairs = downsample(pairs, args.max_per_class, args.seed)

    label2id, id2label = label_maps()
    with open(f"{args.data_dir}/label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    with open(f"{args.data_dir}/id2label.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2, ensure_ascii=False)

    texts = []
    labels = []
    for rec, sup in tqdm(pairs, desc="dataset"):
        texts.append(f"Title: {rec['title']}\nAbstract: {rec['summary']}")
        labels.append(label_id(sup))

    ds = Dataset.from_dict({"text": texts, "label": labels})
    ds = ds.cast_column("label", ClassLabel(num_classes=len(label2id)))

    parts = ds.train_test_split(
        test_size=args.val_fraction,
        seed=args.seed,
        stratify_by_column="label",
    )

    parts["train"].to_json(f"{args.data_dir}/train.json", lines=True)
    parts["test"].to_json(f"{args.data_dir}/validation.json", lines=True)


if __name__ == "__main__":
    main()
