"""Read disabled action-conversion steps in newer official base checkpoints."""

from dataclasses import asdict, dataclass, field

from .pipeline import ProcessorStep, ProcessorStepRegistry


@dataclass
class _DisabledActionConversion(ProcessorStep):
    enabled: bool = False

    def __post_init__(self):
        if self.enabled:
            raise ValueError("This LeRobot port supports only disabled relative/absolute compatibility steps")

    def __call__(self, transition):
        return transition

    def get_config(self):
        return asdict(self)

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class DisabledRelativeActionsProcessorStep(_DisabledActionConversion):
    exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_names: list[str] | None = None


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class DisabledAbsoluteActionsProcessorStep(_DisabledActionConversion):
    pass
