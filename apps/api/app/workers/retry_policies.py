import logging

from google.api_core.exceptions import (
    DeadlineExceeded,
    GatewayTimeout,
    InternalServerError as GoogleInternalServerError,
    ServiceUnavailable,
)
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError as OpenAIInternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_transient_api_error(exception: BaseException) -> bool:
    transient_types: tuple = (
        APIError,
        APITimeoutError,
        RateLimitError,
        APIConnectionError,
        OpenAIInternalServerError,
        ServiceUnavailable,
        DeadlineExceeded,
        GoogleInternalServerError,
        GatewayTimeout,
    )
    return isinstance(exception, transient_types)


agent_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_transient_api_error),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

agent_parent_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception(_is_transient_api_error),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
