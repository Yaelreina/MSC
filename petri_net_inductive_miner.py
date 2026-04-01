"""
Create a Petri net from event log using PM4Py and Inductive Miner.

Uses the Breakfast dataset (ASFormer format) or any XES/CSV event log with PM4Py's
Inductive Miner algorithm to discover a process model as a Petri net.
"""

import os
import pandas as pd
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation import algorithm as evaluation_algo
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.conversion.process_tree import converter as tree_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.util import dataframe_utils


def dataset_to_event_log(dataset_path: str, filter_labels: list[str] | None = None) -> pd.DataFrame:
    """
    Convert ASFormer-style dataset groundTruth to PM4Py event log.

    Each groundTruth .txt file = one case (video). Each line = frame-level action label.
    Consecutive identical labels are collapsed into a single event.

    Args:
        dataset_path: Path to dataset folder (contains groundTruth/, mapping.txt)
        filter_labels: Labels to exclude from traces (e.g. ["SIL", "background"])
    Returns:
        DataFrame with case:concept:name, concept:name, time:timestamp
    """
    gt_dir = os.path.join(dataset_path, "groundTruth")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"groundTruth folder not found at {gt_dir}")

    rows = []
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue
        case_id = fname[:-4]  # e.g. P49_cam02_P49_coffee
        filepath = os.path.join(gt_dir, fname)
        with open(filepath) as f:
            labels = [line.strip() for line in f if line.strip()]

        # Collapse consecutive duplicates
        prev = None
        event_idx = 0
        for frame_idx, label in enumerate(labels):
            if filter_labels and label in filter_labels:
                continue
            if label == prev:
                continue
            prev = label
            rows.append({
                "case:concept:name": case_id,
                "concept:name": label,
                "time:timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=frame_idx),
            })
            event_idx += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No events found. Check groundTruth files.")
    return df


def breakfast_to_event_log(breakfast_path: str, filter_sil: bool = True) -> pd.DataFrame:
    """Backward-compatible wrapper."""
    filters = ["SIL"] if filter_sil else None
    return dataset_to_event_log(breakfast_path, filter_labels=filters)


def _download_url_to_file(url: str, dest_path: str) -> None:
    """Download URL to a file; retry without cert verification if the OS SSL bundle fails (common on macOS)."""
    import ssl
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlretrieve(url, dest_path)
        return
    except (urllib.error.URLError, ssl.SSLError) as e:
        err = str(e).lower()
        if "certificate" not in err and "ssl" not in err:
            raise
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx) as response:
        with open(dest_path, "wb") as out:
            out.write(response.read())


def create_petri_net_from_event_log(
    log_path_or_df,
    output_path: str = "petri_net.png",
    noise_threshold: float = 0.1,
    variant_coverage: float | None = 0.5,
):
    """
    Discover a Petri net from an event log using the Inductive Miner.

    Args:
        log_path_or_df: Path to XES/CSV file or a pandas DataFrame (case:concept:name, concept:name, time:timestamp)
        output_path: Path to save the Petri net visualization
        noise_threshold: 0.0-1.0; higher filters more rare behavior (IMf variant, improves precision)
        variant_coverage: Keep top variants covering this fraction of traces (e.g. 0.8 = 80%). None = no filtering.
    """
    if isinstance(log_path_or_df, pd.DataFrame):
        df = dataframe_utils.convert_timestamp_columns_in_df(log_path_or_df)
        event_log = pm4py.convert_to_event_log(df)
    elif isinstance(log_path_or_df, str):
        log_path = log_path_or_df
        if log_path.endswith(".xes") or log_path.endswith(".xes.gz"):
            event_log = pm4py.read_xes(log_path)
        else:
            df = pm4py.read_csv(log_path)
            df = dataframe_utils.convert_timestamp_columns_in_df(df)
            event_log = pm4py.convert_to_event_log(df)
    else:
        raise TypeError("log_path_or_df must be str or pandas.DataFrame")

    # Optional: keep only top variants covering variant_coverage of traces (improves precision)
    if variant_coverage is not None and 0 < variant_coverage < 1:
        event_log = variants_filter.filter_log_variants_percentage(
            event_log, percentage=variant_coverage
        )
        print(f"  Variant filter: {len(event_log)} traces (top variants covering {variant_coverage*100:.0f}%)")

    # Apply Inductive Miner Infrequent (IMf) - filters rare behavior to improve precision
    process_tree = inductive_miner.apply(
        event_log,
        variant=inductive_miner.Variants.IMf,
        parameters={"noise_threshold": noise_threshold},
    )

    # Convert process tree to Petri net (places, transitions, arcs, initial/final markings)
    net, initial_marking, final_marking = tree_converter.apply(
        process_tree,
        variant=tree_converter.Variants.TO_PETRI_NET
    )

    # Visualize and save
    gviz = pn_visualizer.apply(
        net, initial_marking, final_marking,
        parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
    )
    pn_visualizer.save(gviz, output_path)

    return net, initial_marking, final_marking, event_log


def create_petri_net_from_event_log_heuristics(
    log_path_or_df,
    output_path: str = "petri_net.png",
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.65,
    loop_two_threshold: float = 0.5,
    variant_coverage: float | None = 0.5,
):
    """
    Discover a Petri net from an event log using the Heuristics Miner.

    Args:
        log_path_or_df: Path to XES/CSV file or pandas DataFrame.
        output_path: Path to save the Petri net visualization.
        dependency_threshold: Threshold for dependency relations.
        and_threshold: Threshold for AND-split detection.
        loop_two_threshold: Threshold for length-2 loops.
        variant_coverage: Keep top variants covering this fraction of traces.
    """
    if isinstance(log_path_or_df, pd.DataFrame):
        df = dataframe_utils.convert_timestamp_columns_in_df(log_path_or_df)
        event_log = pm4py.convert_to_event_log(df)
    elif isinstance(log_path_or_df, str):
        log_path = log_path_or_df
        if log_path.endswith(".xes") or log_path.endswith(".xes.gz"):
            event_log = pm4py.read_xes(log_path)
        else:
            df = pm4py.read_csv(log_path)
            df = dataframe_utils.convert_timestamp_columns_in_df(df)
            event_log = pm4py.convert_to_event_log(df)
    else:
        raise TypeError("log_path_or_df must be str or pandas.DataFrame")

    if variant_coverage is not None and 0 < variant_coverage < 1:
        event_log = variants_filter.filter_log_variants_percentage(
            event_log, percentage=variant_coverage
        )
        print(f"  Variant filter: {len(event_log)} traces (top variants covering {variant_coverage*100:.0f}%)")

    try:
        net, initial_marking, final_marking = heuristics_miner.apply(
            event_log,
            parameters={
                "dependency_thresh": dependency_threshold,
                "and_measure_thresh": and_threshold,
                "loop_length_two_thresh": loop_two_threshold,
            },
        )
    except Exception:
        # Fallback to PM4Py high-level API if local heuristics variant differs.
        net, initial_marking, final_marking = pm4py.discover_petri_net_heuristics(
            event_log,
            dependency_threshold=dependency_threshold,
            and_threshold=and_threshold,
            loop_two_threshold=loop_two_threshold,
        )

    gviz = pn_visualizer.apply(
        net, initial_marking, final_marking,
        parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"}
    )
    pn_visualizer.save(gviz, output_path)
    return net, initial_marking, final_marking, event_log


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover a Petri net from an ASFormer dataset or XES/CSV.")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Path to dataset folder (groundTruth/, mapping.txt). Default: Breakfast path if it exists.",
    )
    parser.add_argument(
        "--dataset",
        default="breakfast",
        choices=["breakfast", "50salads"],
        help="Dataset name: sets default silence filter (breakfast→SIL, 50salads→background).",
    )
    parser.add_argument("--output-png", default="petri_net_output.png", help="Path for Petri net PNG visualization.")
    parser.add_argument(
        "--output-pnml",
        default="",
        help="Optional path to save the chosen net as PNML (for ASFormer --petri_mode load).",
    )
    cli = parser.parse_args()

    default_breakfast = "/Users/yaelreina/ASFormer-main/data/breakfast"
    if cli.dataset_root:
        dataset_path = os.path.abspath(os.path.expanduser(cli.dataset_root))
    elif cli.dataset == "50salads":
        dataset_path = os.path.join(os.path.expanduser("~"), "ASFormer-main", "data", "50salads")
    else:
        dataset_path = default_breakfast

    if cli.dataset == "50salads":
        filter_labels = ["background"]
    else:
        filter_labels = ["SIL"]

    OUTPUT_PNG = cli.output_png

    if os.path.isdir(dataset_path):
        print(f"Converting dataset from: {dataset_path}")
        df = dataset_to_event_log(dataset_path, filter_labels=filter_labels)
        print(f"  Loaded {len(df)} events from {df['case:concept:name'].nunique()} traces")
        log_input = df
    elif cli.dataset_root:
        raise SystemExit(
            f"Dataset path does not exist: {dataset_path}\n"
            f"Replace the placeholder with your real folder (must contain groundTruth/ and mapping.txt), e.g.\n"
            f"  --dataset-root \"{os.path.expanduser('~/ASFormer-main/data/50salads')}\""
        )
    else:
        # Fallback only when no --dataset-root was given and default paths are missing: demo XES from PM4Py
        fallback = "running-example.xes"
        if not os.path.exists(fallback):
            url = "https://raw.githubusercontent.com/pm4py/pm4py-core/release/tests/input_data/running-example.xes"
            print(f"No default dataset at {dataset_path}. Downloading sample log...")
            _download_url_to_file(url, fallback)
        log_input = fallback
        print(f"Using: {log_input}")

    # Full log used to evaluate every candidate model.
    if isinstance(log_input, pd.DataFrame):
        full_event_log = pm4py.convert_to_event_log(
            dataframe_utils.convert_timestamp_columns_in_df(log_input)
        )
    else:
        if log_input.endswith(".xes") or log_input.endswith(".xes.gz"):
            full_event_log = pm4py.read_xes(log_input)
        else:
            df_full = pm4py.read_csv(log_input)
            df_full = dataframe_utils.convert_timestamp_columns_in_df(df_full)
            full_event_log = pm4py.convert_to_event_log(df_full)

    # Choose miner: "inductive" or "heuristics".
    miner_type = "heuristics"

    # Grid search for process models that satisfy minimum fitness/precision thresholds.
    TARGET_FITNESS = 0.90
    TARGET_PRECISION = 0.80
    if miner_type == "inductive":
        grid = [
            {"noise_threshold": 0.10, "variant_coverage": 0.90},
            {"noise_threshold": 0.05, "variant_coverage": 0.90},
            {"noise_threshold": 0.00, "variant_coverage": 1.00},
            {"noise_threshold": 0.10, "variant_coverage": 1.00},
            {"noise_threshold": 0.15, "variant_coverage": 0.90},
            {"noise_threshold": 0.05, "variant_coverage": 1.00},
            {"noise_threshold": 0.20, "variant_coverage": 0.95},
            {"noise_threshold": 0.00, "variant_coverage": None},
        ]
    else:
        grid = [
            {"dependency_threshold": 0.80, "and_threshold": 0.80, "loop_two_threshold": 0.70, "variant_coverage": 0.70},
            {"dependency_threshold": 0.85, "and_threshold": 0.85, "loop_two_threshold": 0.80, "variant_coverage": 0.70},
            {"dependency_threshold": 0.90, "and_threshold": 0.85, "loop_two_threshold": 0.80, "variant_coverage": 0.80},
            {"dependency_threshold": 0.95, "and_threshold": 0.90, "loop_two_threshold": 0.90, "variant_coverage": 0.80},
            {"dependency_threshold": 0.90, "and_threshold": 0.90, "loop_two_threshold": 0.90, "variant_coverage": 0.60},
            {"dependency_threshold": 0.85, "and_threshold": 0.80, "loop_two_threshold": 0.80, "variant_coverage": 0.90},
        ]

    chosen = None
    best = None

    print(f"\nSearching Petri net params for target quality (miner={miner_type})...")
    for params in grid:
        print(f"  Try params={params}")
        if miner_type == "inductive":
            net_try, im_try, fm_try, event_log_filtered_try = create_petri_net_from_event_log(
                log_input,
                OUTPUT_PNG,
                noise_threshold=params["noise_threshold"],
                variant_coverage=params["variant_coverage"],
            )
        else:
            net_try, im_try, fm_try, event_log_filtered_try = create_petri_net_from_event_log_heuristics(
                log_input,
                OUTPUT_PNG,
                dependency_threshold=params["dependency_threshold"],
                and_threshold=params["and_threshold"],
                loop_two_threshold=params["loop_two_threshold"],
                variant_coverage=params["variant_coverage"],
            )
        metrics_full_try = evaluation_algo.apply(full_event_log, net_try, im_try, fm_try)
        fitness_full_try = metrics_full_try["fitness"]["log_fitness"]
        precision_full_try = metrics_full_try["precision"]
        fscore_full_try = metrics_full_try.get("fscore", 0.0)

        metrics_subset_try = evaluation_algo.apply(event_log_filtered_try, net_try, im_try, fm_try)
        fitness_subset_try = metrics_subset_try["fitness"]["log_fitness"]
        precision_subset_try = metrics_subset_try["precision"]

        result = {
            "miner_type": miner_type,
            "params": params,
            "net": net_try,
            "im": im_try,
            "fm": fm_try,
            "event_log_filtered": event_log_filtered_try,
            "fitness_full": fitness_full_try,
            "precision_full": precision_full_try,
            "fscore_full": fscore_full_try,
            "fitness_subset": fitness_subset_try,
            "precision_subset": precision_subset_try,
        }
        if best is None or (
            fitness_full_try + precision_full_try > best["fitness_full"] + best["precision_full"]
        ):
            best = result

        print(
            f"    FULL: fitness={fitness_full_try:.4f}, precision={precision_full_try:.4f} | "
            f"SUBSET: fitness={fitness_subset_try:.4f}, precision={precision_subset_try:.4f}"
        )
        if fitness_full_try >= TARGET_FITNESS and precision_full_try >= TARGET_PRECISION:
            chosen = result
            print("    -> Target reached, selecting this model.")
            break

    if chosen is None:
        chosen = best
        print(
            "\nNo candidate reached both thresholds exactly. "
            "Selecting best combined fitness+precision candidate."
        )

    net = chosen["net"]
    im = chosen["im"]
    fm = chosen["fm"]
    event_log_filtered = chosen["event_log_filtered"]
    fitness_full = chosen["fitness_full"]
    precision_full = chosen["precision_full"]
    fscore_full = chosen["fscore_full"]
    fitness_subset = chosen["fitness_subset"]
    precision_subset = chosen["precision_subset"]
    selected_params = chosen["params"]

    print(f"\nPetri net discovered successfully!")
    print(f"  Selected miner: {miner_type}")
    print(f"  Selected params: {selected_params}")
    print(f"  Model built from: {len(event_log_filtered)} traces")
    print(f"  Places: {len(net.places)}, Transitions: {len(net.transitions)}, Arcs: {len(net.arcs)}")
    print(f"\n  On FULL log ({len(full_event_log)} traces) – for disagreement experiment:")
    print(f"    Fitness:  {fitness_full:.4f}  (model captures ~{fitness_full*100:.0f}% of log)")
    print(f"    Precision: {precision_full:.4f}")
    print(f"    F-score:  {fscore_full:.4f}")
    print(f"\n  On TRAINING SUBSET ({len(event_log_filtered)} traces) – process prior quality:")
    print(f"    Fitness:  {fitness_subset:.4f}  Precision: {precision_subset:.4f}  (tight model)")
    print(f"\n  Visualization saved to: {OUTPUT_PNG}")

    if cli.output_pnml:
        from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

        pnml_exporter.apply(net, im, cli.output_pnml, final_marking=fm)
        print(f"\n  PNML saved to: {cli.output_pnml}")

    # Action-to-transition mapper for ASFormer integration
    if isinstance(log_input, pd.DataFrame) and os.path.isdir(dataset_path):
        try:
            from petri_translation_layer import get_mapper_summary

            mapping_path = os.path.join(dataset_path, "mapping.txt")
            summary = get_mapper_summary(net, mapping_path)
            print(f"\n  Action–Transition mapper (for ASFormer):")
            print(f"    Dataset actions: {summary['num_actions']}, net transitions: {summary['num_transitions']}, mapped: {summary['num_mapped']}")
            if summary["unmapped_actions"]:
                print(f"    Unmapped (no transition in net): {summary['unmapped_actions'][:10]}{'...' if len(summary['unmapped_actions']) > 10 else ''}")
        except Exception as e:
            print(f"\n  Mapper build skipped: {e}")


if __name__ == "__main__":
    main()
