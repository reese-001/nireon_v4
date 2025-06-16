# nireon_v4/components/mechanisms/sentinel/constants.py

# Weights are the primary constants that are safe, as they are used as a fallback.
DEFAULT_ALIGNMENT_WEIGHT = 0.4
DEFAULT_FEASIBILITY_WEIGHT = 0.3
DEFAULT_NOVELTY_WEIGHT = 0.3
DEFAULT_WEIGHTS = [DEFAULT_ALIGNMENT_WEIGHT, DEFAULT_FEASIBILITY_WEIGHT, DEFAULT_NOVELTY_WEIGHT]

# Default scores are used as fallbacks during processing errors.
DEFAULT_NOVELTY_SCORE = 5.0
DEFAULT_LLM_SCORE = 5.0

# Deprecated/confusing constants removed:
# DEFAULT_TRUST_THRESHOLD = 0.7  -> Use SentinelMechanismConfig default
# DEFAULT_MIN_AXIS_SCORE = 0.3   -> Use SentinelMechanismConfig default

# Operational constants
MAX_RETRY_ATTEMPTS = 3
ASSESSMENT_TIMEOUT_SECONDS = 30
NOVELTY_SCORE_PRECISION = 4
MAX_BUFFER_SIZE = 100
CACHE_TTL_SECONDS = 3600

# Event names (Good practice to keep these as constants)
EVENT_ASSESSMENT_COMPLETE = 'sentinel.assessment.complete'
EVENT_THRESHOLD_BREACH = 'sentinel.threshold.breach'
EVENT_ADAPTATION_TRIGGERED = 'sentinel.adaptation.triggered'