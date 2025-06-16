from enum import Enum

class EpistemicStage(str, Enum):
    EXPLORATION = 'exploration'   # divergent idea generation
    CRITIQUE    = 'critique'      # evaluation / pruning
    SYNTHESIS   = 'synthesis'     # convergent summarization
    DEFAULT     = 'default'       # fallback / generic

    def __str__(self):
        return self.value
