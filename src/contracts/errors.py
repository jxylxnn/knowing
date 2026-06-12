class ContractError(RuntimeError):
    """Raised when a pipeline contract is violated."""


class ArtifactContractError(ContractError):
    """Raised when trained model artifacts are missing, stale, or incompatible."""


class FeatureSchemaContractError(ContractError):
    """Raised when runtime features do not match the trained feature schema."""


class ProjectionSchemaContractError(ContractError):
    """Raised when exported projection CSV schema is incompatible with the query layer."""


class ScheduleContractError(ContractError):
    """Raised when schedule scraper output is malformed."""
