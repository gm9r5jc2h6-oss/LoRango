from __future__ import annotations

from typing import Any


def _index_modules(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    modules = inventory.get("modules")
    if not isinstance(modules, list):
        raise ValueError("inventory must contain a modules list")
    return {str(row["name"]): row for row in modules}


def _materialize_roles(
    role_templates: dict[str, str], gate_block: int, action_block: int | None
) -> dict[str, str]:
    result = {}
    for role, template in sorted(role_templates.items()):
        result[role] = template.format(
            block=gate_block,
            gate_block=gate_block,
            action_block=action_block if action_block is not None else gate_block,
        )
    return result


def generate_candidates(
    architecture: dict[str, Any], inventory: dict[str, Any]
) -> dict[str, Any]:
    by_name = _index_modules(inventory)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for template in architecture["candidate_templates"]:
        gate_blocks = [int(value) for value in template["block_ids"]]
        action_blocks_raw = template.get("action_block_ids")
        for gate_block in gate_blocks:
            action_blocks = (
                [int(value) for value in action_blocks_raw]
                if action_blocks_raw is not None
                else [None]
            )
            for action_block in action_blocks:
                if action_block is not None:
                    minimum_distance = int(template.get("minimum_action_distance", 0))
                    if action_block - gate_block < minimum_distance:
                        continue
                roles = _materialize_roles(template["roles"], gate_block, action_block)
                missing = [path for path in roles.values() if path not in by_name]
                unsupported = [
                    path
                    for path in roles.values()
                    if path in by_name and not bool(by_name[path].get("exportable", False))
                ]
                suffix = f"b{gate_block}"
                if action_block is not None:
                    suffix += f"_to_b{action_block}"
                candidate_id = f"{template['id']}_{suffix}"
                if missing or unsupported:
                    rejected.append(
                        {
                            "candidate_id": candidate_id,
                            "missing_modules": missing,
                            "unsupported_modules": unsupported,
                        }
                    )
                    continue
                parameter_width = sum(
                    int(by_name[path]["in_features"]) + int(by_name[path]["out_features"])
                    for path in roles.values()
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "family": template["family"],
                        "template_id": template["id"],
                        "gate_block": gate_block,
                        "action_block": action_block,
                        "roles": roles,
                        "module_shapes": {
                            role: by_name[path]["weight_shape"] for role, path in roles.items()
                        },
                        "parameter_width_per_rank": parameter_width,
                        "exportable": True,
                    }
                )

    candidates.sort(key=lambda row: row["candidate_id"])
    rejected.sort(key=lambda row: row["candidate_id"])
    return {
        "schema_version": 1,
        "architecture_id": architecture["architecture_id"],
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "candidates": candidates,
        "rejected": rejected,
    }

