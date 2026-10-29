"""Stage orchestration. Scripts are thin wrappers around these functions so tests can call them directly.

    preprocess_patient : DICOM -> seg -> SUV -> resample -> mask transfer -> crop -> masking -> MA-MIP (cached)
    extract_features   : MIP views -> frozen backbone -> aggregation -> features.parquet
    train_eval         : features + cohort + folds -> per-model out-of-fold predictions -> metrics
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fdgmip.cache import Cache
from fdgmip.config import Config
from fdgmip.domain import Volume
from fdgmip.eval import compare_models, summarize
from fdgmip.features.pet_stats import pet_stats
from fdgmip.io import list_series, read_series
from fdgmip.io.nifti import read_nifti, write_volume
from fdgmip.preprocess import (check_inclusion, compute_suv, crop, liver_normalize, mask_out_organs, reorient,
                               resample, transfer_masks)
from fdgmip.preprocess.suv import extract_params
from fdgmip.projection import compose_views
from fdgmip.projection.mip import rotation_center
from fdgmip.registry import (AGGREGATORS, BACKBONES, CROPS, HARMONIZERS, MODELS, PROJECTORS, SEGMENTERS,
                             import_all)

log = logging.getLogger(__name__)
import_all()

SUV_KEY_SECTIONS = ["preprocess.orientation", "preprocess.spacing_mm", "preprocess.suv.scan_time_tag"]
SEG_KEY_SECTIONS = ["segmentation", "preprocess.orientation"]
MIP_KEY_SECTIONS = ["preprocess", "segmentation", "crop", "organ_masking", "projection"]


# ---------------------------------------------------------------- preprocess
def preprocess_patient(cfg: Config, patient_id: str, patient_dir: Path, ct_uid: str, pet_uid: str,
                       segmenter, cache: Cache) -> dict:
    series = {s.uid: s for s in list_series(patient_dir)}
    try:
        ct_s, pet_s = series[ct_uid], series[pet_uid]
    except KeyError as e:
        return {"patient_id": patient_id, "excluded": f"series {e} not found"}
    incl = cfg.section("inclusion")
    viol = check_inclusion(ct_s, incl.get("ct", {}) or {}) + check_inclusion(pet_s, incl.get("pet", {}) or {})
    if viol:
        return {"patient_id": patient_id, "excluded": "; ".join(viol)}

    orient = cfg.get("preprocess.orientation")
    organs = list(cfg.get("segmentation.organs"))

    # CT -> organ masks (cached as NIfTI on the CT grid)
    seg_dir = cache.dir(patient_id, "seg", cfg.hash(SEG_KEY_SECTIONS, extra=ct_uid))
    if all((seg_dir / f"{o}.nii.gz").exists() for o in organs):
        masks = {o: read_nifti(seg_dir / f"{o}.nii.gz", "MASK", ct_s.frame_of_reference_uid) for o in organs}
    else:
        ct = reorient(read_series(ct_s), orient)
        masks = segmenter.segment(ct, organs)
        for o, m in masks.items():
            write_volume(m, seg_dir / f"{o}.nii.gz")

    # PET -> SUV, isotropic (cached)
    suv_path = cache.dir(patient_id, "suv", cfg.hash(SUV_KEY_SECTIONS, extra=pet_uid)) / "suv.nii.gz"
    if suv_path.exists():
        suv = read_nifti(suv_path, "PT", pet_s.frame_of_reference_uid)
    else:
        pet = reorient(read_series(pet_s), orient)
        suv = compute_suv(pet, extract_params(pet.meta["header"], cfg.get("preprocess.suv.scan_time_tag")))
        suv = resample(suv, tuple(cfg.get("preprocess.spacing_mm")), "linear")
        write_volume(suv.replace(meta={}), suv_path)

    masks_pet = transfer_masks(masks, suv)  # raises on FrameOfReferenceUID mismatch
    if cfg.get("preprocess.suv.liver_normalize"):
        suv = liver_normalize(suv, masks_pet[cfg.get("preprocess.suv.liver_organ")])

    stats = pet_stats(suv, masks_pet[cfg.get("features.pet_stats.voi_organ")], cfg.get("features.pet_stats.mtv_threshold"))

    # crop -> organ masking -> multi-angle MIP (cached)
    mip_dir = cache.dir(patient_id, "mip", cfg.hash(MIP_KEY_SECTIONS, extra=[ct_uid, pet_uid]))
    mip_path = mip_dir / "views.npy"
    if not mip_path.exists():
        fill = float(cfg.get("crop.fill_value"))
        box = CROPS.build(cfg.get("crop.strategy")).box(suv, masks_pet)
        vol = crop(suv, box, fill)
        cmasks = {k: crop(m, box, 0) for k, m in masks_pet.items()}
        om = cfg.section("organ_masking")
        if om["enabled"]:
            vol = mask_out_organs(vol, cmasks, om["organs"], float(om["fill_value"]))
        projector = PROJECTORS.build(cfg.get("projection.projector"))
        center = rotation_center(vol, cmasks, cfg.get("projection.center"), cfg.get("projection.center_organ"))
        views = compose_views(vol, cmasks, projector, center, cfg.get("projection.channels"))
        np.save(mip_path, views)
        (mip_dir / "box.json").write_text(json.dumps({"lo": box.lo, "hi": box.hi, "angles": projector.angles()}))
    return {"patient_id": patient_id, "excluded": None, "mip_path": str(mip_path),
            **{f"pet_{k}": v for k, v in stats.items()}}


def run_preprocess(cfg: Config) -> pd.DataFrame:
    sel = pd.read_csv(cfg.get("paths.series_selection"), dtype=str)
    root = Path(cfg.get("paths.dicom_root"))
    cache = Cache(cfg.get("paths.derived_root"))
    segmenter = SEGMENTERS.build(cfg.get("segmentation.segmenter"))
    rows = []
    for r in sel.itertuples(index=False):
        log.info("preprocess %s", r.patient_id)
        try:
            rows.append(preprocess_patient(cfg, r.patient_id, root / r.patient_id, r.ct_series_uid, r.pet_series_uid,
                                           segmenter, cache))
        except Exception as e:  # keep going; failures are data for the exclusion log
            log.exception("patient %s failed", r.patient_id)
            rows.append({"patient_id": r.patient_id, "excluded": f"error: {type(e).__name__}: {e}"})
    manifest = pd.DataFrame(rows)
    out = cache.root / "manifests" / f"manifest_{cfg.hash(MIP_KEY_SECTIONS)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    log.info("manifest -> %s (%d included, %d excluded)", out, manifest.excluded.isna().sum(), manifest.excluded.notna().sum())
    return manifest


# ---------------------------------------------------------------- features
def extract_features(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    backbone = BACKBONES.build(cfg.get("features.backbone"))
    agg = AGGREGATORS.build(cfg.get("features.aggregator"))
    rows = []
    for r in manifest[manifest.excluded.isna()].itertuples(index=False):
        views = np.load(r.mip_path)
        vec = agg.aggregate(backbone.extract(views))
        row = {"patient_id": r.patient_id, **{k: v for k, v in r._asdict().items() if k.startswith("pet_")}}
        row.update({f"mip_{i:04d}": float(v) for i, v in enumerate(vec)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


def run_features(cfg: Config, manifest_path: str | Path) -> Path:
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    feats = extract_features(cfg, manifest)
    out = Path(cfg.get("paths.derived_root")) / "features" / cfg.hash(MIP_KEY_SECTIONS + ["features"]) / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out)
    log.info("features (%d patients x %d cols) -> %s", *feats.shape, out)
    return out


# ---------------------------------------------------------------- train / eval
def build_design_matrix(cfg: Config, features: pd.DataFrame, cohort: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cc = cfg.section("cohort")
    clin = cohort[list(cc["clinical_columns"])].add_prefix("clin_")
    extra = cohort[list(cc.get("extra_columns", []) or [])]
    X = features.join(clin, how="inner").join(extra, how="inner")
    missing = cohort.index.difference(X.index)
    if len(missing):
        log.warning("%d cohort patients have no features (excluded/failed): %s", len(missing), list(missing)[:20])
    return X, cohort.loc[X.index, cc["label_column"]]


def train_eval(cfg: Config, X: pd.DataFrame, y: pd.Series, folds: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    folds = folds[folds.patient_id.isin(X.index)]
    harm = cfg.section("harmonization")
    batch_col = harm.get("batch_column")
    mip_cols = [c for c in X.columns if c.startswith("mip_")]
    preds = []
    for name, spec in cfg.section("models").items():
        spec = {"name": "tabular", **spec}
        if harm["stage"] == "model":
            spec.update(harmonization={k: v for k, v in harm.items() if k not in ("stage", "batch_column")}, batch_column=batch_col)
        for (rep, k), g in folds.groupby(["repeat", "fold"]):
            test_ids = g.patient_id.to_numpy()
            train_ids = folds[(folds["repeat"] == rep) & ~folds.patient_id.isin(test_ids)].patient_id.to_numpy()
            Xtr, Xte = X.loc[train_ids].copy(), X.loc[test_ids].copy()
            if harm["stage"] == "feature" and harm["name"] != "none":
                h = HARMONIZERS.build({kk: v for kk, v in harm.items() if kk not in ("stage", "batch_column")})
                h.fit(Xtr[mip_cols], Xtr[batch_col])                       # train fold only
                Xtr[mip_cols] = h.transform(Xtr[mip_cols], Xtr[batch_col])
                Xte[mip_cols] = h.transform(Xte[mip_cols], Xte[batch_col])
            model = MODELS.build(spec)
            p = model.fit(Xtr, y.loc[train_ids].to_numpy()).predict_proba(Xte)
            preds.append(pd.DataFrame({"patient_id": test_ids, "repeat": rep, "fold": k, "model": name,
                                       "y": y.loc[test_ids].to_numpy(), "p": p}))
    preds_df = pd.concat(preds, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_dir / "preds.csv", index=False)
    ev = cfg.section("eval")
    thr = np.arange(*ev["dca_thresholds"])
    metrics, dca = summarize(preds_df, ev["bootstrap_n"], cfg.get("seed"), thr)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    dca.to_csv(out_dir / "decision_curve.csv", index=False)
    cmp = compare_models(preds_df, ev["compare"]["reference"], ev["compare"]["candidates"])
    cmp.to_csv(out_dir / "comparison.csv", index=False)
    log.info("\n%s", metrics[metrics.metric == "auroc"].pivot(index="model", columns="repeat", values="value"))
    return preds_df
