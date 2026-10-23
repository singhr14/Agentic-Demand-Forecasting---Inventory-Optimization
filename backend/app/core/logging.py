import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
        timestamp=True,
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class RequestIdFilter(logging.Filter):
    def __init__(self, request_id: str = "-") -> None:
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = self.request_id
        return True


def extra(request_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
    payload = {"request_id": request_id or "-"}
    payload.update(kwargs)
    return payload
