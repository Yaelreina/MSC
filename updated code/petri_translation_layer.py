"""
Translation layer: map Breakfast action classes to PM4Py Petri net transitions.

Bridges the 48 Breakfast actions (ASFormer labels) to Transition objects in the net,
so the Transformer's predictions can be translated into process-state updates.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any

# PM4Py Transition type (avoid hard dependency at import if only using dicts)
try:
    from pm4py.objects.petri_net.obj import PetriNet
    _Transition = PetriNet.Transition
except ImportError:
    _Transition = Any


def load_breakfast_actions(mapping_path: str) -> Tuple[Dict[int, str], Dict[str, int], List[str]]:
    """
    Load the 48 Breakfast action classes from mapping.txt.

    Args:
        mapping_path: Path to breakfast/mapping.txt (lines: "id action_name")

    Returns:
        action_id_to_name: {0: "SIL", 1: "pour_cereals", ...}
        action_name_to_id: {"pour_cereals": 1, ...}
        action_names_ordered: ["SIL", "pour_cereals", ...] index = action_id
    """
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"Breakfast mapping not found: {mapping_path}")

    action_id_to_name: Dict[int, str] = {}
    action_name_to_id: Dict[str, int] = {}
    action_names_ordered: List[str] = []

    with open(mapping_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            idx = int(parts[0])
            name = parts[1]
            action_id_to_name[idx] = name
            action_name_to_id[name] = idx
            action_names_ordered.append(name)

    return action_id_to_name, action_name_to_id, action_names_ordered


def build_action_to_transition_mapper(
    net: "PetriNet",
    mapping_path: str,
) -> Dict[str, Optional[_Transition]]:
    """
    Build a dictionary mapping each Breakfast action name to its Transition in the net.

    When the Transformer predicts an action (e.g. "crack_egg"), this mapper tells you
    which transition in the Petri net should fire to update the process state.

    Args:
        net: PM4Py Petri net (from create_petri_net_from_event_log).
        mapping_path: Path to breakfast/mapping.txt.

    Returns:
        action_name -> Transition or None. None means that action has no corresponding
        transition in the net (e.g. SIL, or an action filtered out by the miner).
    """
    _, _, action_names_ordered = load_breakfast_actions(mapping_path)

    # Build label -> transition from net (only visible transitions have labels)
    label_to_transition: Dict[str, _Transition] = {}
    for t in net.transitions:
        if t.label is not None:
            # PM4Py can have duplicate labels in some nets; last one wins (or use first)
            label_to_transition[t.label] = t

    # Map every Breakfast action to a transition or None
    action_to_transition: Dict[str, Optional[_Transition]] = {}
    for name in action_names_ordered:
        action_to_transition[name] = label_to_transition.get(name)

    return action_to_transition


def build_action_id_to_transition_mapper(
    net: "PetriNet",
    mapping_path: str,
) -> Tuple[Dict[int, Optional[_Transition]], Dict[str, Optional[_Transition]]]:
    """
    Same as build_action_to_transition_mapper but also returns mapping by action ID (0..47).

    Returns:
        action_id_to_transition: {0: None, 1: <Transition pour_cereals>, ...}
        action_name_to_transition: {"pour_cereals": <Transition>, ...}
    """
    id_to_name, _, _ = load_breakfast_actions(mapping_path)
    name_to_transition = build_action_to_transition_mapper(net, mapping_path)
    id_to_transition = {
        aid: name_to_transition.get(name) for aid, name in id_to_name.items()
    }
    return id_to_transition, name_to_transition


def build_transition_to_index(net: "PetriNet") -> Dict[_Transition, int]:
    """
    Assign a stable integer index to each visible transition (for tensor embedding).

    Returns:
        transition -> index in [0, num_visible_transitions). Order is by transition name.
    """
    visible = sorted([t for t in net.transitions if t.label is not None], key=lambda t: t.name)
    return {t: i for i, t in enumerate(visible)}


def get_mapper_summary(
    net: "PetriNet",
    mapping_path: str,
) -> Dict[str, Any]:
    """
    Build all mappers and return a summary (for sanity check and export).

    Returns:
        dict with:
          - action_name_to_transition
          - action_id_to_transition
          - transition_to_index
          - action_id_to_transition_index (action_id -> tensor index, or -1 if no transition)
          - unmapped_actions: list of action names with no transition in the net
          - num_actions, num_transitions, num_mapped
    """
    id_to_trans, name_to_trans = build_action_id_to_transition_mapper(net, mapping_path)
    trans_to_idx = build_transition_to_index(net)

    # action_id -> transition index for tensor (e.g. -1 for SIL or unmapped)
    action_id_to_transition_index: Dict[int, int] = {}
    unmapped: List[str] = []
    id_to_name, _, _ = load_breakfast_actions(mapping_path)

    for aid, name in id_to_name.items():
        t = name_to_trans.get(name)
        if t is None:
            action_id_to_transition_index[aid] = -1
            unmapped.append(name)
        else:
            action_id_to_transition_index[aid] = trans_to_idx[t]

    return {
        "action_name_to_transition": name_to_trans,
        "action_id_to_transition": id_to_trans,
        "transition_to_index": trans_to_idx,
        "action_id_to_transition_index": action_id_to_transition_index,
        "unmapped_actions": unmapped,
        "num_actions": len(id_to_name),
        "num_transitions": len(trans_to_idx),
        "num_mapped": len(id_to_name) - len(unmapped),
    }


if __name__ == "__main__":
    # Example: build net then mapper (run from same dir as petri_net_inductive_miner.py)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from petri_net_inductive_miner import breakfast_to_event_log, create_petri_net_from_event_log

    BREAKFAST_PATH = "/Users/yaelreina/ASFormer-main/data/breakfast"
    MAPPING_PATH = os.path.join(BREAKFAST_PATH, "mapping.txt")

    df = breakfast_to_event_log(BREAKFAST_PATH, filter_sil=True)
    net, im, fm, _ = create_petri_net_from_event_log(df, "petri_net_output.png", noise_threshold=0.75, variant_coverage=0.5)
    summary = get_mapper_summary(net, MAPPING_PATH)

    print("Action → Transition mapper summary:")
    print(f"  Actions: {summary['num_actions']}, Transitions in net: {summary['num_transitions']}, Mapped: {summary['num_mapped']}")
    print(f"  Unmapped: {summary['unmapped_actions']}")
    print("\n  action_id_to_transition_index (for PyTorch tensor): first 12 keys → transition index or -1")
    for aid in list(summary["action_id_to_transition_index"].keys())[:12]:
        print(f"    action_id {aid} → transition_index {summary['action_id_to_transition_index'][aid]}")
