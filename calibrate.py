"""
calibrate.py
---------------
STEP 3 of the pipeline: extract embeddings from every image (real +
synthetic) and fit the real decay constant k and noise growth rate
gamma, per Section 11 of the main model document.

Each person's SMALLEST real captured depth is used as their reference
(E_ref) for each view. Every other depth (real or synthetic) is
compared back to that reference via cosine similarity, and a curve is
fit through the resulting (depth, similarity) points.

Usage:
    python calibrate.py --dataset_root dataset --output_dir calibration_output

Requires: enroll_capture.py to have been run first.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from utils import VIEWS, extract_embedding_and_width, cosine_sim, parse_depth_folder, list_people, list_depth_folders

# Depths below this are treated as "near range" -- real indoor distances
# you can actually walk and test at home. Depths at or above this are
# assumed to be far-range (synthetic/far-range data, if you generated any).
# This matches the cutoff used elsewhere to pick
# the real reference image.
NEAR_RANGE_MAX_M = 10.0


def collect_embeddings(dataset_root: Path):
    """Nested dict: embeddings[person_id][depth][view] = np.ndarray
    Also returns widths[person_id][depth][view] = face width in pixels."""
    embeddings = {}
    widths = {}
    for person_dir in list_people(dataset_root):
        person_id = person_dir.name
        embeddings[person_id] = {}
        widths[person_id] = {}
        for depth_dir in list_depth_folders(person_dir):
            depth = parse_depth_folder(depth_dir.name)
            embeddings[person_id][depth] = {}
            widths[person_id][depth] = {}
            for view in VIEWS:
                candidates = list(depth_dir.glob(f"{view}.*"))
                if not candidates:
                    continue
                result = extract_embedding_and_width(candidates[0])
                if result is not None:
                    emb, width_px = result
                    embeddings[person_id][depth][view] = emb
                    widths[person_id][depth][view] = width_px
                else:
                    print(f"  [warn] no face detected: {candidates[0]}")
        n_depths = len(embeddings[person_id])
        print(f"Processed {person_id}: {n_depths} depth folders")
    return embeddings, widths


def compute_similarity_table(embeddings):
    """Each row is (depth_offset, similarity, absolute_target_depth).
    Keeping the absolute target depth lets us separately fit a curve
    using ONLY near-range (real, <NEAR_RANGE_MAX_M) comparisons, which
    is what actually matters for indoor testing -- mixing in far-range
    synthetic points would bias the fit toward distances you're not
    testing at."""
    rows = []
    per_person_rows = {}
    reference_depths = {}
    for person_id, depths in embeddings.items():
        if not depths:
            continue
        ref_depth = min(depths.keys())  # smallest captured depth = reference
        reference_depths[person_id] = ref_depth
        ref_views = depths[ref_depth]
        per_person_rows[person_id] = []
        for depth, views in depths.items():
            if depth == ref_depth:
                continue
            for view, emb in views.items():
                if view not in ref_views:
                    continue
                sim = cosine_sim(emb, ref_views[view])
                rows.append((depth - ref_depth, sim, depth))
                per_person_rows[person_id].append((depth - ref_depth, sim, depth))
    return rows, per_person_rows, reference_depths


def decay_model(x, k):
    """x = depth - reference_depth. sim = exp(-k*x)"""
    return np.exp(-k * np.asarray(x))


def fit_k(rows):
    xs = np.array([r[0] for r in rows], dtype=float)
    sims = np.array([r[1] for r in rows], dtype=float)
    popt, pcov = curve_fit(decay_model, xs, sims, p0=[0.015], bounds=(0, 1))
    k_fit = popt[0]
    k_stderr = float(np.sqrt(pcov[0][0]))
    return k_fit, k_stderr, xs, sims


def estimate_gamma(rows, k_fit):
    xs = np.array([r[0] for r in rows], dtype=float)
    sims = np.array([r[1] for r in rows], dtype=float)
    predicted = decay_model(xs, k_fit)
    residual = np.abs(predicted - sims)

    def noise_model(x, sigma0, gamma):
        return sigma0 * (1 + gamma * np.asarray(x))

    try:
        popt, _ = curve_fit(noise_model, xs, residual, p0=[0.05, 0.02], bounds=(0, [1, 1]))
        return float(popt[0]), float(popt[1])
    except RuntimeError:
        print("  [warn] gamma fit did not converge, returning defaults")
        return 0.05, 0.02


def plot_fit(xs, sims, k_fit, out_path, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_grid = np.linspace(0, xs.max(), 200)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(xs, sims, alpha=0.5, label="observed similarity (all people/views)")
    plt.plot(x_grid, decay_model(x_grid, k_fit), color="red", linewidth=2, label=f"fitted decay, k={k_fit:.5f}")
    plt.xlabel("Depth beyond each person's reference (m)")
    plt.ylabel("Cosine similarity to reference")
    plt.title(f"Calibration: fitted decay curve{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved calibration plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="calibration_output")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting embeddings from {dataset_root} ...")
    embeddings, widths = collect_embeddings(dataset_root)

    print("Computing similarity table vs each person's reference...")
    rows, per_person_rows, reference_depths = compute_similarity_table(embeddings)
    if len(rows) < 5:
        raise RuntimeError(
            "Not enough data points to fit a decay curve. "
            "Make sure enroll_capture.py have both been run."
        )

    # average reference face width per person (front view, at their reference depth)
    reference_widths = {}
    for person_id, ref_depth in reference_depths.items():
        w = widths.get(person_id, {}).get(ref_depth, {})
        if w:
            reference_widths[person_id] = float(np.mean(list(w.values())))

    print("Fitting global k (all data: real + synthetic far-range)...")
    k_fit, k_stderr, xs, sims = fit_k(rows)
    print(f"  global_k = {k_fit:.5f}  (stderr {k_stderr:.5f})")

    print("Estimating noise growth gamma (all data)...")
    sigma0, gamma = estimate_gamma(rows, k_fit)
    print(f"  global sigma0 = {sigma0:.4f}, global gamma = {gamma:.4f}")

    # --- NEAR-RANGE FIT: real data only, < NEAR_RANGE_MAX_M ---
    # This is the fit that actually matters for indoor testing. If you
    # walk 0.5m-4m in your room, THIS k/gamma is what api_server.py and
    # a distance-trend test should use, since the global fit above is dominated
    # by far-range synthetic points that barely move at these distances.
    near_rows_2 = [(o, s) for (o, s, ad) in rows if ad < NEAR_RANGE_MAX_M]
    near_k = near_k_stderr = near_sigma0 = near_gamma = None
    near_xs = near_sims = None
    if len(near_rows_2) >= 5:
        print(f"\nFitting NEAR-RANGE k (real data only, {len(near_rows_2)} points < {NEAR_RANGE_MAX_M:.0f}m)...")
        near_k, near_k_stderr, near_xs, near_sims = fit_k(near_rows_2)
        print(f"  near_k = {near_k:.5f}  (stderr {near_k_stderr:.5f})")
        near_sigma0, near_gamma = estimate_gamma(near_rows_2, near_k)
        print(f"  near_sigma0 = {near_sigma0:.4f}, near_gamma = {near_gamma:.4f}")
    else:
        print(f"\n  [warn] only {len(near_rows_2)} near-range (<{NEAR_RANGE_MAX_M:.0f}m) datapoints found.")
        print("  Capture at least 3 real depths per person (e.g. --depth 0.5, 1.5, 4.0)")
        print("  for a meaningful near-range fit. Falling back to the global fit for now.")

    print("\nFitting per-person k_i (all data)...")
    per_person_k = {}
    for person_id, p_rows in per_person_rows.items():
        p_rows_2 = [(o, s) for (o, s, ad) in p_rows]
        if len(p_rows_2) < 3:
            continue
        k_i, k_i_stderr, _, _ = fit_k(p_rows_2)
        per_person_k[person_id] = {"k": k_i, "stderr": k_i_stderr}
        print(f"  {person_id}: k_i = {k_i:.5f}")

    results = {
        "global_k": k_fit,
        "global_k_stderr": k_stderr,
        "sigma0": sigma0,
        "gamma": gamma,
        # near_k/near_gamma/near_sigma0 are what api_server.py and
        # scripts that consume this JSON will actually use (they fall back to the
        # global_* values above if near-range data isn't available).
        "near_k": near_k if near_k is not None else k_fit,
        "near_k_stderr": near_k_stderr,
        "near_sigma0": near_sigma0 if near_sigma0 is not None else sigma0,
        "near_gamma": near_gamma if near_gamma is not None else gamma,
        "num_near_datapoints": len(near_rows_2),
        "near_range_max_m": NEAR_RANGE_MAX_M,
        "reference_depths_per_person": reference_depths,
        "reference_widths_px_per_person": reference_widths,
        "num_datapoints": len(rows),
        "per_person_k": per_person_k,
        "note": "x-axis in this fit is (depth - each person's own reference depth), "
                "since reference depths may differ slightly across people in a WFH setup. "
                "'near_*' fields are fit on real close-range data only and are what live "
                "matching actually uses; 'global_*' fields include far-range synthetic data."
    }
    results_path = output_dir / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration results to {results_path}")

    # Save the actual reference embeddings as the enrollment gallery
    gallery = {}
    for person_id, ref_depth in reference_depths.items():
        for view, emb in embeddings[person_id][ref_depth].items():
            gallery[f"{person_id}__{view}"] = emb
    gallery_path = output_dir / "gallery.npz"
    np.savez(gallery_path, **gallery)
    print(f"Saved enrollment gallery ({len(gallery)} reference embeddings) to {gallery_path}")

    plot_fit(xs, sims, k_fit, output_dir / "decay_fit.png", title_suffix=" (global: real + synthetic)")
    if near_xs is not None:
        plot_fit(near_xs, near_sims, near_k, output_dir / "near_decay_fit.png",
                  title_suffix=" (near-range: real data only)")

    print("\nDone. api_server.py will use 'near_k' / 'near_gamma' /")
    print("'near_sigma0' from calibration_results.json (falling back to the global_*")
    print("values only if not enough near-range data was found).")


if __name__ == "__main__":
    main()
