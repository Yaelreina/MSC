"""
Dynamic marking vector: track token positions as the video (trace) progresses.

Produces, for every frame, a vector of length num_places: 1 if a token is in that place, 0 otherwise.
Handles "disagreement" when the observed action is not enabled in the current marking (~0.79 fitness).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pm4py.objects.petri_net.obj import PetriNet, Marking
    from pm4py.objects.petri_net import semantics
except ImportError as e:
    raise ImportError("pm4py required for dynamic_marking") from e


def get_places_ordered(net: PetriNet) -> List[PetriNet.Place]:
    """Stable order of places (by name) for consistent vector indices."""
    return sorted(net.places, key=lambda p: p.name)


def marking_to_vector(marking: Marking, places_ordered: List[PetriNet.Place]) -> np.ndarray:
    """
    Convert a PM4Py Marking to a binary vector of length len(places_ordered).

    Args:
        marking: current marking (Place -> token count).
        places_ordered: fixed order of places (e.g. from get_places_ordered(net)).

    Returns:
        Vector of shape (num_places,) with 1 if place has >= 1 token, else 0.
    """
    n = len(places_ordered)
    out = np.zeros(n, dtype=np.float32)
    for i, p in enumerate(places_ordered):
        if marking.get(p, 0) >= 1:
            out[i] = 1.0
    return out


def compute_dynamic_marking_vectors(
    net: PetriNet,
    initial_marking: Marking,
    frame_actions: List[str],
    action_name_to_transition: Dict[str, Optional[PetriNet.Transition]],
    use_weak_execute_on_disagreement: bool = False,
    silence_label: str = "SIL",
) -> Tuple[np.ndarray, List[bool]]:
    """
    For every frame, produce the marking vector after applying the process up to that frame.

    Fires a transition only when the action *changes* (new segment), so we do not double-fire
    for the same action across consecutive frames. When the observed action is not enabled
    (disagreement), marking is either kept (strict) or updated via weak_execute (lenient).

    Args:
        net: PM4Py Petri net.
        initial_marking: initial marking.
        frame_actions: list of length num_frames; frame_actions[f] = action name at frame f.
        action_name_to_transition: from petri_translation_layer (action name -> Transition or None).
        use_weak_execute_on_disagreement: if True, on disagreement use weak_execute; else keep marking.
        silence_label: label to skip (no transition, no fire).

    Returns:
        marking_vectors: (num_frames, num_places) float32, 0/1.
        disagreement_per_frame: (num_frames,) bool, True if at that frame we had a disagreement.
    """
    places_ordered = get_places_ordered(net)
    num_places = len(places_ordered)
    num_frames = len(frame_actions)

    # Marking state; we only update when action changes (new segment)
    current_marking = copy.copy(initial_marking)
    prev_action: Optional[str] = None
    marking_vectors = np.zeros((num_frames, num_places), dtype=np.float32)
    disagreement_per_frame = [False] * num_frames

    for f in range(num_frames):
        action = frame_actions[f]
        # Fire only on segment change (new action)
        if action != prev_action:
            prev_action = action
            if action and action != silence_label:
                trans = action_name_to_transition.get(action)
                if trans is not None:
                    if semantics.is_enabled(trans, net, current_marking):
                        new_m = semantics.execute(trans, net, current_marking)
                        if new_m is not None:
                            current_marking = new_m
                    else:
                        # Disagreement: video says this action, net says it's not enabled
                        disagreement_per_frame[f] = True
                        if use_weak_execute_on_disagreement:
                            current_marking = semantics.weak_execute(trans, current_marking)
                # else: action has no transition in net (e.g. filtered out), keep marking

        marking_vectors[f] = marking_to_vector(current_marking, places_ordered)

    return marking_vectors, disagreement_per_frame


def compute_dynamic_marking_vectors_from_ids(
    net: PetriNet,
    initial_marking: Marking,
    frame_action_ids: List[int],
    action_id_to_transition: Dict[int, Optional[PetriNet.Transition]],
    action_id_to_name: Optional[Dict[int, str]] = None,
    use_weak_execute_on_disagreement: bool = False,
    silence_id: int = 0,
) -> Tuple[np.ndarray, List[bool]]:
    """
    Same as compute_dynamic_marking_vectors but input is frame_action_ids (0..47) per frame.

    action_id_to_transition: from get_mapper_summary()["action_id_to_transition"] (id -> Transition or None).
    silence_id: Breakfast SIL = 0.
    """
    if action_id_to_name is None:
        # Default: assume 0=SIL, rest map to transition by id if present
        action_id_to_name = {i: f"action_{i}" for i in range(48)}
        action_id_to_name[0] = "SIL"
    frame_actions = [action_id_to_name.get(aid, "SIL") for aid in frame_action_ids]
    name_to_trans = {
        action_id_to_name[aid]: t for aid, t in action_id_to_transition.items()
    }
    return compute_dynamic_marking_vectors(
        net,
        initial_marking,
        frame_actions,
        name_to_trans,
        use_weak_execute_on_disagreement=use_weak_execute_on_disagreement,
        silence_label="SIL",
    )


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from petri_net_inductive_miner import breakfast_to_event_log, create_petri_net_from_event_log
    from petri_translation_layer import get_mapper_summary, load_breakfast_actions

    BREAKFAST_PATH = "/Users/yaelreina/ASFormer-main/data/breakfast"
    MAPPING_PATH = os.path.join(BREAKFAST_PATH, "mapping.txt")
    GT_DIR = os.path.join(BREAKFAST_PATH, "groundTruth")

    # Build net and mappers
    df = breakfast_to_event_log(BREAKFAST_PATH, filter_sil=True)
    net, initial_marking, _, _ = create_petri_net_from_event_log(
        df, "petri_net_output.png", noise_threshold=0.75, variant_coverage=0.5
    )
    summary = get_mapper_summary(net, MAPPING_PATH)
    action_name_to_transition = summary["action_name_to_transition"]
    num_places = len(get_places_ordered(net))
    print(f"Net has {num_places} places. Generating dynamic marking for one video.")

    # One video: read frame-level labels
    sample = next(f for f in sorted(os.listdir(GT_DIR)) if f.endswith(".txt"))
    with open(os.path.join(GT_DIR, sample)) as f:
        frame_actions = [line.strip() for line in f if line.strip()]
    marking_vectors, disagreement = compute_dynamic_marking_vectors(
        net, initial_marking, frame_actions, action_name_to_transition
    )
    print(f"Video {sample}: {len(frame_actions)} frames, marking_vectors shape {marking_vectors.shape}")
    print(f"Disagreements (frames where action not enabled): {sum(disagreement)}")
    print(f"First frame marking (first 8 places): {marking_vectors[0][:8]}")
