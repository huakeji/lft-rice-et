"""Station membership guards; independent of model code and data loading.

This guards early stopping and fitted station sets only. It does not certify
historical hyperparameter selection, raw timestamps, or feature provenance.
"""

from datetime import date


def sequence_within_split(target_dates, split):
    """All seven physical target dates must belong to their declared period."""
    bounds = {
        "train": (date.min, date(2018, 12, 31)),
        "val": (date(2019, 1, 1), date(2019, 12, 31)),
        "test": (date(2020, 1, 1), date(2020, 12, 31)),
    }
    if split not in bounds or len(target_dates) != 7:
        return False
    days = [date.fromisoformat(str(x)[:10]) for x in target_dates]
    lo, hi = bounds[split]
    return all(lo <= d <= hi for d in days) and all(
        (b - a).days == 1 for a, b in zip(days, days[1:])
    )


def resolve_validation_stations(job):
    explicit = job.validation_stations
    validation = list(job.eval_stations if explicit is None else explicit)
    if not validation:
        raise ValueError("Validation station list must not be empty")
    if job.spatial_selection_guard:
        if explicit is None:
            raise ValueError("Spatial selection requires explicit validation_stations")
        train, heldout = set(job.train_stations), set(job.eval_stations)
        if not train or not heldout or train & heldout:
            raise ValueError("Spatial train and held-out stations must be nonempty and disjoint")
        if not set(validation) <= train:
            raise ValueError("Spatial validation stations must come only from training stations")
        for name in ("vocab_stations", "norm_stations"):
            fitted = set(getattr(job, name))
            if not fitted or not fitted <= train:
                raise ValueError(f"{name} must be nonempty and restricted to training stations")
        if job.val_split != "val" or tuple(job.train_splits or ("train",)) != ("train",):
            raise ValueError("Spatial selection requires train fitting and val early stopping")
    return validation
