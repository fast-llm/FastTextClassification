


from enum import Enum
from transformers.utils import ExplicitEnum

class ParallelMode(Enum):
    DISTRIBUTED = "distributed"
    SINGLE = "single"
    NOT_PARALLEL = "not_parallel"

class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
    ONE_CYCLE_LR = 'OneCycleLR'

class LossFnType(ExplicitEnum):
    CROSS_ENTROPY = "CrossEntropyLoss"
    BCE = "BCELoss"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"
    NLL = "NLLLoss"
    POISSON_NLL = "PoissonNLLLoss"
    KL_DIV = "KLDivLoss"
