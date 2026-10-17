"""Bounded second-lane replay of the legacy final PFT training recipe.

This is a retrospective development run on the current corrected 2017 panel.  It
does not alter or share outputs with the primary candidate-search queue.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from research_exploratory_worker import validate
from research_fold_data import DevelopmentData, FoldData
from research_spatial_data import SpatialFoldData
from research_spatial_membership import verify as verify_spatial_membership
from research_models import ForecastModel
from research_run_manifest import digest
from run_conditional_batch import atomic_json
from matched_baselines import Encoder, NUM, CATS


R = Path(__file__).resolve().parents[2]
DEFAULT_OUT = R / "reports/v11_audit_20260909/research_route_40/parallel_legacy_pft_current2017_seed0"
MANIFEST = R / "configs/research_route_40/exploratory_comparison_v1.json"
CONFIG = {
    "candidate": "legacy_final_recipe",
    "width": 128,
    "layers": 2,
    "heads": 4,
    "dropout": 0.1,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "optimizer": "Adam",
    "scheduler": "cosine_warmup",
    "warmup_epochs": 5,
    "batch_size": 1024,
    "max_epochs": 200,
    "patience": 25,
}


class BoundedStop(RuntimeError):
    pass


def atomic_torch(value, path):
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(value, temporary)
    os.replace(temporary, path)


def encoder_from_state(state):
    """Restore the training-only encoder so a retry does not rescan all shards."""
    encoder = Encoder()
    encoder.ni = [NUM.index(name) for name in state["numeric_names"]]
    encoder.ci = [CATS.index(name) for name in state["categorical_names"]]
    encoder.mean = np.asarray(state["mean"], dtype=float)
    encoder.scale = np.asarray(state["scale"], dtype=float)
    encoder.vocab = {CATS.index(name): values for name, values in state["vocabulary"].items()}
    if encoder.state() != state:
        raise ValueError("Restored preprocessing state changed")
    return encoder


def main(args):
    started = time.monotonic()
    out = Path(args.out).resolve()
    if not out.is_relative_to(R):
        raise ValueError("Project-local output required")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection_record.json").exists():
        print("Verified replay already complete; no duplicate fit", flush=True)
        return 0
    training = out / "training"
    training.mkdir(exist_ok=True)
    manifest_path = Path(args.manifest).resolve()
    manifest, protocol, shards, data_sha = validate(manifest_path)
    cfg = dict(CONFIG)
    cfg["threads"] = args.threads
    cfg["seconds"] = args.seconds
    config_sha = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    binding = {
        "model": "pft",
        "seed": args.seed,
        "fold_id": manifest["fold_id"],
        "data_sha256": data_sha,
        "protocol_sha256": digest(R / "configs/research_route_40/protocol.json"),
        "config_sha256": config_sha,
        "implementation_sha256": digest(Path(__file__)),
        "study_arm": "legacy_final_training_recipe_on_current_development_panel",
        "scientific_status": "EXPLORATORY_UNVERIFIED_TIME_SEMANTICS",
        "manifest_path": str(manifest_path.relative_to(R)),
    }
    if bool(args.spatial_membership) != (args.spatial_fold is not None):
        raise ValueError("Spatial membership and fold must be supplied together")
    membership = None
    if args.spatial_membership:
        membership_path = Path(args.spatial_membership).resolve()
        geometry = verify_spatial_membership(membership_path, 100.0)
        with membership_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        membership = {row["station"]: row["role"] for row in rows if int(row["fold"]) == args.spatial_fold}
        if len(membership) != geometry["stations"]:
            raise ValueError("Spatial membership cohort changed")
        binding.update({
            "fold_id": f"conditional_spatial_{args.spatial_fold}_temporal_{manifest['outer_year']}",
            "study_arm": "conditional_legacy_cohort_100km_spatial_temporal_pft",
            "spatial_fold": args.spatial_fold,
            "spatial_membership_sha256": digest(membership_path),
            "spatial_geometry_status": geometry["status"],
        })
    amendment = R / "configs/research_route_40/budget_amendment_same_day_20260914.json"
    if manifest["fold_id"] == "exploratory_same_day_temporal_2018":
        if digest(amendment) != amendment.with_suffix(".sha256").read_text().strip():
            raise ValueError("Same-day budget amendment changed")
        binding["budget_amendment_sha256"] = digest(amendment)
    torch.set_num_threads(args.threads)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)

    def guard():
        if time.monotonic() - started > args.seconds:
            raise BoundedStop("parallel lane time limit")
        if psutil.disk_usage(R).free < protocol["resources"]["min_disk_bytes"]:
            raise BoundedStop("disk headroom")
        if psutil.virtual_memory().available < 1024**3:
            raise BoundedStop("system available memory below 1 GiB")
        if psutil.Process().memory_info().rss > 6 * 1024**3:
            raise BoundedStop("parallel lane RSS above 6 GiB")

    try:
        fold_data = SpatialFoldData(
            shards, manifest["outer_year"], membership, manifest.get("target_offset", 0)
        ) if membership is not None else FoldData(
            shards,
            manifest["outer_year"],
            manifest["excluded_stations"],
            manifest.get("target_offset", 0),
        )
        data = DevelopmentData(fold_data)
        config_path = out / "config.json"
        last_path = training / "last.pt"
        if last_path.exists() and config_path.exists():
            saved_config = json.loads(config_path.read_text())
            if saved_config["binding"] != binding:
                raise ValueError("Replay preprocessing binding mismatch")
            encoder = encoder_from_state(saved_config["preprocessing"])
            target = saved_config["target"]
        else:
            encoder, target = data.fit_encoder()
            atomic_json(config_path, {
                "binding": binding,
                "configuration": cfg,
                "preprocessing": encoder.state(),
                "target": target,
                "scientific_status": binding["scientific_status"],
                "outer_evaluation_performed": False,
                "budget_reservation_seconds": args.seconds,
                "target_offset": manifest.get("target_offset", 0),
            })
        guard()
        example = next(data.encoded_batches(encoder, "train", 1))[0]
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = ForecastModel("pft", encoder.state(), example.shape[-1], CONFIG)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=CONFIG["warmup_epochs"]
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["max_epochs"] - CONFIG["warmup_epochs"], eta_min=0.0
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[CONFIG["warmup_epochs"]]
        )
        start_epoch = 0
        best = float("inf")
        best_epoch = None
        stale = 0
        history = []
        if last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            if state["binding"] != binding:
                raise ValueError("Replay checkpoint binding mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            start_epoch = state["epoch"] + 1
            best = state["best"]
            best_epoch = state["best_epoch"]
            stale = state["stale"]
            history = state["history"]

        for epoch in range(start_epoch, CONFIG["max_epochs"]):
            if stale >= CONFIG["patience"]:
                break
            guard()
            model.train()
            train_sum = 0.0
            train_n = 0
            for x, y, _, _ in data.encoded_batches(encoder, "train", CONFIG["batch_size"]):
                guard()
                order = np.random.permutation(len(y))
                xx = torch.as_tensor(x[order], dtype=torch.float32)
                yy = torch.as_tensor((y[order] - target["mean"]) / target["scale"], dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(xx) - yy).square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite replay loss")
                loss.backward()
                optimizer.step()
                train_sum += float(loss.detach()) * yy.numel()
                train_n += yy.numel()
            model.eval()
            sse = 0.0
            count = 0
            with torch.no_grad():
                for x, y, _, _ in data.encoded_batches(encoder, "val", CONFIG["batch_size"]):
                    guard()
                    prediction = model.predict(torch.as_tensor(x, dtype=torch.float32)).numpy()
                    prediction = prediction * target["scale"] + target["mean"]
                    sse += float(np.square(prediction.astype(float) - y).sum())
                    count += y.size
            rmse = (sse / count) ** 0.5
            lr = optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, "training_standardized_mse": train_sum / train_n, "validation_rmse": rmse, "lr": lr})
            if rmse < best:
                best = rmse
                best_epoch = epoch
                stale = 0
                atomic_torch({"binding": binding, "model": model.state_dict(), "epoch": epoch, "validation_rmse": rmse}, training / "best.pt")
            else:
                stale += 1
            scheduler.step()
            atomic_torch({
                "binding": binding,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best": best,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            }, last_path)
            atomic_json(out / "progress.json", {"status": "RUNNING", "epoch": epoch, "best_epoch": best_epoch, "best_validation_rmse": best, "elapsed_seconds": time.monotonic() - started})

        checkpoint = torch.load(training / "best.pt", map_location="cpu", weights_only=False)
        if checkpoint["binding"] != binding:
            raise ValueError("Best replay checkpoint binding mismatch")
        model.load_state_dict(checkpoint["model"])
        model.eval()
        prediction_dir = out / "validation_predictions"
        prediction_dir.mkdir(exist_ok=True)
        parts = []
        count = 0
        sse = 0.0
        with torch.no_grad():
            for index, (x, y, origin, station) in enumerate(data.encoded_batches(encoder, "val", CONFIG["batch_size"])):
                guard()
                prediction = model.predict(torch.as_tensor(x, dtype=torch.float32)).numpy()
                prediction = prediction * target["scale"] + target["mean"]
                if not np.isfinite(prediction).all():
                    raise ValueError("Invalid replay prediction")
                part = prediction_dir / f"part_{index:05d}.npz"
                np.savez_compressed(part, prediction=prediction, target=y, origin=origin, station=station)
                parts.append({"path": str(part.relative_to(out)), "sha256": digest(part), "sequences": len(y)})
                count += y.size
                sse += float(np.square(prediction.astype(float) - y).sum())
        replay_rmse = (sse / count) ** 0.5
        if not np.isclose(replay_rmse, best, rtol=1e-6, atol=1e-7):
            raise ValueError("Best replay checkpoint metric mismatch")
        atomic_json(out / "validation_predictions.json", {"parts": parts, "target_count": count, "pooled_rmse": replay_rmse})
        record = {**binding, "status": "COMPLETE", "metric_role": "internal_validation", "validation_rmse": replay_rmse, "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters())}
        atomic_json(out / "selection_record.json", record)
        atomic_json(out / "status.json", {"status": "COMPLETE", "validation_rmse": best, "best_epoch": best_epoch, "epochs_ran": len(history), "active_seconds": time.monotonic() - started})
        return 0
    except BoundedStop as exc:
        atomic_json(out / "status.json", {"status": "RESOURCE_STOP", "reason": str(exc), "active_seconds": time.monotonic() - started})
        return 2
    except BaseException as exc:
        atomic_json(out / "status.json", {"status": "FAILED", "reason": repr(exc), "active_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--spatial-membership")
    parser.add_argument("--spatial-fold", type=int, choices=range(5))
    raise SystemExit(main(parser.parse_args()))
