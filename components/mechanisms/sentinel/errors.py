# C:\Users\erees\Documents\development\nireon_staging\nireon\application\mechanisms\sentinel\errors.py
class NireonApplicationError(Exception):
    pass
class SentinelError(NireonApplicationError):
    pass
class SentinelAssessmentError(SentinelError):
    pass
class SentinelScoringError(SentinelError):
    pass
class SentinelLLMParsingError(SentinelError):
    pass
