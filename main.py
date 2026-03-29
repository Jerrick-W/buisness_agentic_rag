"""Entry point for running the application directly."""

import uvicorn

from app.config import validate_settings

if __name__ == "__main__":
    settings = validate_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
