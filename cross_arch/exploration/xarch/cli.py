from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_record
from .backend import inventory_model
from .candidates import generate_candidates
from .config import load_architecture, load_policy
from .plan import build_plan
from .scoring import rank_candidates, select_candidates
from .util import atomic_write_json, import_symbol, load_json, load_jsonl


def _bundle_root() -> Path:
    return Path(__file__).resolve().parent.parent


def command_inventory(args: argparse.Namespace) -> None:
    loader = import_symbol(args.loader)
    loader_config = load_json(args.loader_config) if args.loader_config else {}
    model = loader(loader_config)
    document = inventory_model(model)
    document["loader"] = args.loader
    atomic_write_json(args.output, document)


def command_generate(args: argparse.Namespace) -> None:
    architecture = load_architecture(args.architecture)
    result = generate_candidates(architecture, load_json(args.inventory))
    atomic_write_json(args.output, result)


def command_rank(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    result = rank_candidates(load_json(args.candidates), load_jsonl(args.metrics), policy)
    atomic_write_json(args.output, result)


def command_select(args: argparse.Namespace) -> None:
    policy = load_policy(args.policy)
    result = select_candidates(load_json(args.ranked), policy)
    atomic_write_json(args.output, result)


def command_plan(args: argparse.Namespace) -> None:
    architecture = load_architecture(args.architecture)
    policy = load_policy(args.policy)
    atomic_write_json(
        args.output,
        build_plan(architecture, policy, args.architecture, args.policy, args.data),
    )


def command_audit(args: argparse.Namespace) -> None:
    result = audit_record(load_json(args.record))
    atomic_write_json(args.output, result)
    if result["status"] != "PASS":
        raise SystemExit(2)


def command_demo(args: argparse.Namespace) -> None:
    root = _bundle_root()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture_path = root / "architectures" / "synthetic_demo.json"
    policy_path = root / "policies" / "controlled_v1.json"
    data_path = root / "data" / "harmless_pairs.json"
    architecture = load_architecture(architecture_path)
    policy = load_policy(policy_path)
    candidates = generate_candidates(
        architecture, load_json(root / "examples" / "synthetic_inventory.json")
    )
    ranked = rank_candidates(
        candidates,
        load_jsonl(root / "examples" / "synthetic_probe_metrics.jsonl"),
        policy,
    )
    selected = select_candidates(ranked, policy)
    plan = build_plan(architecture, policy, architecture_path, policy_path, data_path)
    atomic_write_json(args.output_dir / "candidates.json", candidates)
    atomic_write_json(args.output_dir / "ranked_candidates.json", ranked)
    atomic_write_json(args.output_dir / "selected_topology.json", selected)
    atomic_write_json(args.output_dir / "run_plan.json", plan)
    if selected["status"] != "selected":
        raise RuntimeError("synthetic demo unexpectedly produced a negative result")
    winner = selected["selected"][0]
    expected = "joint_attention_b2_to_b5"
    if winner["candidate_id"] != expected or winner["rank"] != 4:
        raise RuntimeError(f"unexpected demo winner: {winner['candidate_id']} rank {winner['rank']}")
    print(json.dumps({"status": "PASS", "winner": f"{expected}_r4"}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--loader", required=True)
    inventory.add_argument("--loader-config", type=Path)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(function=command_inventory)

    generate = subparsers.add_parser("generate-candidates")
    generate.add_argument("--architecture", type=Path, required=True)
    generate.add_argument("--inventory", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.set_defaults(function=command_generate)

    rank = subparsers.add_parser("rank")
    rank.add_argument("--policy", type=Path, required=True)
    rank.add_argument("--candidates", type=Path, required=True)
    rank.add_argument("--metrics", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.set_defaults(function=command_rank)

    select = subparsers.add_parser("select")
    select.add_argument("--policy", type=Path, required=True)
    select.add_argument("--ranked", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(function=command_select)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--architecture", type=Path, required=True)
    plan.add_argument("--policy", type=Path, required=True)
    plan.add_argument("--data", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(function=command_plan)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--record", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(function=command_audit)

    demo = subparsers.add_parser("demo")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.set_defaults(function=command_demo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)

