"""
pet_abnormal_omni_test.py
=========================
pet_abnormal_omni_best.pth 로 학습된 모델의 테스트 스크립트.

기능:
  1. Skin / Eyes 테스트셋 전체 평가 (Accuracy, Precision, Recall, F1, Confusion Matrix)
  2. 클래스별 상세 성능 리포트 (Classification Report)
  3. 유사 클래스 혼동률 분석 (Similar-group confusion)
  4. predict_anomaly() 앙상블 함수 단위 테스트
  5. 결과 PNG 저장 (Confusion Matrix 시각화)

실행 예시:
  python pet_abnormal_omni_test.py
  python pet_abnormal_omni_test.py --ckpt my_model.pth --skin_root files/4_Animal_Skin --eyes_root files/5_Animal_Eyes
"""

import os
import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ──────────────────────────────────────────────────────────────────────────────
# pet_abnormal_omni 의 클래스·모델 정의를 그대로 재사용
# ──────────────────────────────────────────────────────────────────────────────
from pet_abnormal_omni_train import (
    AnomalyMultiBackbone,
    AnomalyDataset,
    predict_anomaly,
    SKIN_CLASSES,
    EYES_CLASSES,
    EYES_SIMILAR_GROUPS,
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    NUM_IMAGES_PER_SAMPLE,
)

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_CKPT      = "pet_abnormal_omni_best.pth"
DEFAULT_SKIN_ROOT = "files/4_Animal_Skin"
DEFAULT_EYES_ROOT = "files/5_Animal_Eyes"
OUTPUT_DIR        = "test_results"

# ──────────────────────────────────────────────────────────────────────────────
# 1. 모델 로드
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str) -> tuple[AnomalyMultiBackbone, list, list]:
    """
    체크포인트를 로드하고 (model, skin_classes, eyes_classes) 반환.
    체크포인트에 클래스 정보가 없으면 소스코드 상수를 사용.
    """
    print(f"\n📂 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    skin_classes = ckpt.get("skin_classes", SKIN_CLASSES)
    eyes_classes = ckpt.get("eyes_classes", EYES_CLASSES)

    model = AnomalyMultiBackbone(len(skin_classes), len(eyes_classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(DEVICE)

    print(f"  ✅ Loaded  (trained epoch: {ckpt.get('epoch', '?')} | "
          f"best avg acc: {ckpt.get('best_avg_acc', 0)*100:.2f}%)")
    print(f"  Skin classes ({len(skin_classes)}): {skin_classes}")
    print(f"  Eyes classes ({len(eyes_classes)}): {eyes_classes}")
    return model, skin_classes, eyes_classes


# ──────────────────────────────────────────────────────────────────────────────
# 2. 단일 task 평가
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_task(
    model: AnomalyMultiBackbone,
    root_dir: str,
    class_names: list,
    task: str,          # "skin" | "eyes"
) -> dict:
    """
    root_dir 하위의 클래스 폴더 전체를 테스트셋으로 평가.

    Returns dict with keys:
        acc, precision, recall, f1,
        all_preds, all_labels,
        class_names, report_str
    """
    if not os.path.isdir(root_dir):
        print(f"  ⚠️  {root_dir} not found, skipping {task} evaluation.")
        return {}

    dataset = AnomalyDataset(root_dir, class_names, task=task, is_train=False)
    if len(dataset) == 0:
        print(f"  ⚠️  No samples found in {root_dir}")
        return {}

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )

    all_preds, all_labels = [], []

    print(f"\n🔍 Evaluating [{task.upper()}]  ({len(dataset)} samples, {len(class_names)} classes)")
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"  {task}", leave=False):
            images = images.to(DEVICE)
            logits = model(images, task=task)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    report_str = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%  Recall: {recall*100:.2f}%  F1: {f1*100:.2f}%")

    return {
        "acc"        : acc,
        "precision"  : precision,
        "recall"     : recall,
        "f1"         : f1,
        "all_preds"  : all_preds,
        "all_labels" : all_labels,
        "class_names": class_names,
        "report_str" : report_str,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. 유사 클래스 혼동 분석
# ──────────────────────────────────────────────────────────────────────────────

def analyze_similar_group_confusion(
    all_labels: np.ndarray,
    all_preds:  np.ndarray,
    class_names: list,
    similar_groups: list,
) -> dict:
    """
    Eyes 유사 클래스 쌍(상/하, 초기/성숙 등) 내 혼동률을 계산.
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    results = {}

    print("\n📊 Similar-group Confusion Analysis (Eyes):")
    for group in similar_groups:
        idxs  = [name_to_idx[n] for n in group if n in name_to_idx]
        if len(idxs) < 2:
            continue

        # 해당 그룹 클래스 중 하나를 정답으로 가진 샘플
        mask         = np.isin(all_labels, idxs)
        group_labels = all_labels[mask]
        group_preds  = all_preds[mask]

        total = len(group_labels)
        # 정답은 그룹 내이지만 다른 클래스로 예측한 경우
        intra_confused = np.sum(
            np.isin(group_preds, idxs) & (group_preds != group_labels)
        )
        inter_confused = np.sum(~np.isin(group_preds, idxs))
        correct        = np.sum(group_preds == group_labels)

        confusion_rate = intra_confused / total if total > 0 else 0.0
        label_str = " / ".join(group)
        print(f"  [{label_str}]")
        print(f"    Total: {total}  Correct: {correct}  "
              f"Intra-confused: {intra_confused}  Inter-confused: {inter_confused}")
        print(f"    Intra-confusion rate: {confusion_rate*100:.2f}%")
        results[label_str] = {
            "total": total, "correct": correct,
            "intra_confused": int(intra_confused),
            "inter_confused": int(inter_confused),
            "intra_confusion_rate": confusion_rate,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. Confusion Matrix 시각화
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    all_labels: np.ndarray,
    all_preds:  np.ndarray,
    class_names: list,
    task: str,
    save_path: str,
) -> None:
    cm = confusion_matrix(all_labels, all_preds)
    # row-normalize for readability
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    n = len(class_names)
    fig_size = max(10, n * 0.7)
    fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2 + 2, fig_size))

    for ax, data, title, fmt in [
        (axes[0], cm,      f"{task.upper()} Confusion Matrix (count)", "d"),
        (axes[1], cm_norm, f"{task.upper()} Confusion Matrix (normalized)", ".2f"),
    ]:
        im = ax.imshow(data, interpolation="nearest",
                       cmap="Blues" if fmt == "d" else "YlOrRd")
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        short_names = [c.replace("dog_", "D:").replace("cat_", "C:") for c in class_names]
        ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, fontsize=11, fontweight="bold")

        thresh = data.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                txt = f"{val:{fmt}}" if val > 0 else ""
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved confusion matrix → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. predict_anomaly() 앙상블 단위 테스트
# ──────────────────────────────────────────────────────────────────────────────

def run_ensemble_spot_check(
    model: AnomalyMultiBackbone,
    skin_root: str,
    eyes_root: str,
    skin_classes: list,
    eyes_classes: list,
    n_samples: int = 5,
    n_images:  int = NUM_IMAGES_PER_SAMPLE,
) -> None:
    """
    각 task에서 임의 클래스를 골라 n_images 장 앙상블 예측을 수행하고
    predict_anomaly() 출력 포맷 및 pet_type 마스킹을 검증.
    """
    print(f"\n🧪 Ensemble Spot-Check (n_samples={n_samples}, n_images={n_images})")

    for task, root, classes in [
        ("skin", skin_root,  skin_classes),
        ("eyes", eyes_root, eyes_classes),
    ]:
        if not os.path.isdir(root):
            continue

        available = [
            c for c in classes
            if os.path.isdir(os.path.join(root, c))
        ]
        sampled_classes = np.random.choice(available,
                                           size=min(n_samples, len(available)),
                                           replace=False)

        print(f"\n  [{task.upper()}]")
        for cls in sampled_classes:
            cls_dir   = os.path.join(root, cls)
            pet_type  = "dog" if cls.startswith("dog_") else "cat"
            img_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not img_files:
                continue

            chosen = np.random.choice(img_files,
                                      size=min(n_images, len(img_files)),
                                      replace=len(img_files) < n_images)
            images = [Image.open(os.path.join(cls_dir, f)).convert("RGB")
                      for f in chosen]

            result = predict_anomaly(
                model=model,
                images=images,
                task=task,
                pet_type=pet_type,
                class_names=classes,
                device=DEVICE,
            )

            correct_str = "✅" if result["predicted_class"] == cls else "❌"
            print(f"    {correct_str} GT: {cls:<35}  "
                  f"Pred: {result['predicted_class']:<35}  "
                  f"Conf: {result['confidence']*100:.1f}%")
            print(f"         Top-3: " +
                  "  |  ".join(f"{n}({p*100:.1f}%)" for n, p in result["top3"]))

            # 검증: pet_type 이 다른 클래스가 top3에 없어야 함
            wrong_type = [n for n, _ in result["top3"]
                          if not n.startswith(pet_type + "_")]
            if wrong_type:
                print(f"    ⚠️  [WARN] pet_type mask 실패: {wrong_type}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. 결과 요약 JSON 저장
# ──────────────────────────────────────────────────────────────────────────────

def save_summary(
    skin_result: dict,
    eyes_result: dict,
    similar_confusion: dict,
    out_path: str,
) -> None:
    summary = {}
    for task, res in [("skin", skin_result), ("eyes", eyes_result)]:
        if res:
            summary[task] = {
                "accuracy" : round(res["acc"],       4),
                "precision": round(res["precision"], 4),
                "recall"   : round(res["recall"],    4),
                "f1"       : round(res["f1"],        4),
            }
    summary["similar_group_confusion"] = similar_confusion
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved summary JSON → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. 클래스별 성능 바 차트
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1(
    all_labels: np.ndarray,
    all_preds:  np.ndarray,
    class_names: list,
    task: str,
    save_path: str,
) -> None:
    _, _, f1_scores, support = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    short = [c.replace("dog_", "D:").replace("cat_", "C:") for c in class_names]
    colors = ["steelblue" if c.startswith("dog_") else "tomato" for c in class_names]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.7), 5))
    bars = ax.bar(short, f1_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class"); ax.set_ylabel("F1 Score")
    ax.set_title(f"{task.upper()} – Per-Class F1 Score", fontweight="bold")
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=55, ha="right", fontsize=8)
    ax.axhline(np.mean(f1_scores), color="gray", linestyle="--",
               linewidth=1, label=f"Mean F1 = {np.mean(f1_scores):.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # annotate support count on each bar
    for bar, sup in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                str(sup), ha="center", va="bottom", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾 Saved per-class F1 chart → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="pet_abnormal_omni 테스트")
    p.add_argument("--ckpt",       default=DEFAULT_CKPT,
                   help=f"체크포인트 경로 (default: {DEFAULT_CKPT})")
    p.add_argument("--skin_root",  default=DEFAULT_SKIN_ROOT,
                   help=f"Skin 데이터 루트 (default: {DEFAULT_SKIN_ROOT})")
    p.add_argument("--eyes_root",  default=DEFAULT_EYES_ROOT,
                   help=f"Eyes 데이터 루트 (default: {DEFAULT_EYES_ROOT})")
    p.add_argument("--out_dir",    default=OUTPUT_DIR,
                   help=f"결과 저장 폴더 (default: {OUTPUT_DIR})")
    p.add_argument("--spot_n",     type=int, default=5,
                   help="앙상블 spot-check 샘플 수 (default: 5)")
    p.add_argument("--no_spot",    action="store_true",
                   help="앙상블 spot-check 생략")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("  pet_abnormal_omni  —  Test Evaluation")
    print("=" * 65)
    print(f"  Device   : {DEVICE}")
    print(f"  Ckpt     : {args.ckpt}")
    print(f"  Skin root: {args.skin_root}")
    print(f"  Eyes root: {args.eyes_root}")
    print(f"  Output   : {args.out_dir}")

    # ── 1. 모델 로드 ──────────────────────────────────────────────────────────
    model, skin_classes, eyes_classes = load_model(args.ckpt)

    # ── 2. Skin 평가 ──────────────────────────────────────────────────────────
    skin_result = evaluate_task(model, args.skin_root, skin_classes, task="skin")

    # ── 3. Eyes 평가 ──────────────────────────────────────────────────────────
    eyes_result = evaluate_task(model, args.eyes_root, eyes_classes, task="eyes")

    # ── 4. Classification Report 출력 & 저장 ──────────────────────────────────
    for task, res in [("skin", skin_result), ("eyes", eyes_result)]:
        if not res:
            continue
        print(f"\n{'='*65}")
        print(f"  {task.upper()} Classification Report")
        print(f"{'='*65}")
        print(res["report_str"])
        report_path = os.path.join(args.out_dir, f"{task}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"[{task.upper()} Classification Report]\n\n")
            f.write(res["report_str"])
        print(f"  💾 Saved → {report_path}")

    # ── 5. Confusion Matrix 시각화 ────────────────────────────────────────────
    for task, res in [("skin", skin_result), ("eyes", eyes_result)]:
        if not res:
            continue
        cm_path = os.path.join(args.out_dir, f"{task}_confusion_matrix.png")
        plot_confusion_matrix(
            res["all_labels"], res["all_preds"],
            res["class_names"], task, cm_path
        )
        f1_path = os.path.join(args.out_dir, f"{task}_per_class_f1.png")
        plot_per_class_f1(
            res["all_labels"], res["all_preds"],
            res["class_names"], task, f1_path
        )

    # ── 6. 유사 클래스 혼동 분석 (Eyes 전용) ─────────────────────────────────
    similar_confusion = {}
    if eyes_result:
        similar_confusion = analyze_similar_group_confusion(
            eyes_result["all_labels"],
            eyes_result["all_preds"],
            eyes_result["class_names"],
            EYES_SIMILAR_GROUPS,
        )

    # ── 7. 앙상블 Spot-Check ──────────────────────────────────────────────────
    if not args.no_spot:
        run_ensemble_spot_check(
            model,
            args.skin_root,
            args.eyes_root,
            skin_classes,
            eyes_classes,
            n_samples=args.spot_n,
        )

    # ── 8. 최종 요약 ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "test_summary.json")
    save_summary(skin_result, eyes_result, similar_confusion, summary_path)

    print(f"\n{'='*65}")
    print("  ✅  Evaluation Complete")
    print(f"{'='*65}")
    if skin_result:
        print(f"  Skin  → Acc: {skin_result['acc']*100:.2f}%  "
              f"F1: {skin_result['f1']*100:.2f}%")
    if eyes_result:
        print(f"  Eyes  → Acc: {eyes_result['acc']*100:.2f}%  "
              f"F1: {eyes_result['f1']*100:.2f}%")
    if skin_result and eyes_result:
        avg_acc = (skin_result["acc"] + eyes_result["acc"]) / 2
        print(f"  Avg Acc: {avg_acc*100:.2f}%")
    print(f"\n  📁 All results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
