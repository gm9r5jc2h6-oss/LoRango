from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def resolve_module(root: Any, path: str) -> Any:
    current = root
    for component in path.split("."):
        if component.isdigit():
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value
    if isinstance(value, (tuple, list)):
        for child in value:
            try:
                return _first_tensor(child)
            except TypeError:
                pass
    raise TypeError("hook value contains no tensor-like object")


def tensor_summary(value: Any) -> dict[str, Any]:
    tensor = _first_tensor(value).detach().float()
    return {
        "shape": [int(v) for v in tensor.shape],
        "dtype": str(getattr(value, "dtype", "unknown")),
        "rms": float(tensor.square().mean().sqrt().cpu()),
        "abs_max": float(tensor.abs().max().cpu()),
        "finite": bool(tensor.isfinite().all().cpu()),
    }


class ActivationRecorder:
    """Collect per-call module input/output summaries using temporary hooks."""

    def __init__(self, root: Any, module_paths: Iterable[str]) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.handles = []
        for path in sorted(set(module_paths)):
            module = resolve_module(root, path)

            def hook(_module: Any, args: Any, output: Any, *, module_path: str = path) -> None:
                self.rows[module_path].append(
                    {
                        "call_index": len(self.rows[module_path]),
                        "input": tensor_summary(args),
                        "output": tensor_summary(output),
                    }
                )

            self.handles.append(module.register_forward_hook(hook))

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "modules": [
                {"name": name, "calls": calls}
                for name, calls in sorted(self.rows.items())
            ],
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

