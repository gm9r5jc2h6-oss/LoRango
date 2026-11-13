from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RolloutState:
    latent: Any
    timestep: Any
    image_ids: Any = None
    metadata: Mapping[str, Any] | None = None


class ArchitectureBackend(ABC):
    """Required model-specific boundary for a real exploration run.

    The control plane deliberately does not guess these semantics. A backend
    must expose ordinary frozen-model operations and a standard static export;
    it must not hide selection logic inside the backend.
    """

    @abstractmethod
    def load_frozen_model(self, model_config: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def encode_prompts(self, prompts: Mapping[str, str]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare_initial_state(self, seed: int, resolution: tuple[int, int]) -> RolloutState:
        raise NotImplementedError

    @abstractmethod
    def timesteps(self, count: int) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        state: RolloutState,
        conditioning: Any,
        enabled_branches: frozenset[str],
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def scheduler_step(self, state: RolloutState, prediction: Any) -> RolloutState:
        raise NotImplementedError

    @abstractmethod
    def attach_probe(self, candidate: Mapping[str, Any], rank: int, seed: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def trainable_probe_parameters(self, probe: Any) -> Iterable[Any]:
        raise NotImplementedError

    @abstractmethod
    def export_two_files(self, probe: Any, destination: str) -> Mapping[str, str]:
        raise NotImplementedError

    @abstractmethod
    def unload_development_hooks(self) -> None:
        raise NotImplementedError


def inventory_model(model: Any) -> dict[str, Any]:
    """Inventory two-dimensional weighted modules without importing torch.

    This works for PyTorch modules because it relies only on `named_modules`,
    `weight.ndim`, and `weight.shape`. A model-specific loader can be passed to
    the CLI as `package.module:function`.
    """

    modules = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            continue
        shape = [int(value) for value in weight.shape]
        modules.append(
            {
                "name": name,
                "class": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "in_features": shape[1],
                "out_features": shape[0],
                "weight_shape": shape,
                "exportable": True,
            }
        )
    return {"schema_version": 1, "module_count": len(modules), "modules": modules}

