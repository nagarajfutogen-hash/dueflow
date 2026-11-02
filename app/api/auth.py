import secrets

from fastapi import HTTPException, Request, status


async def require_token(request: Request) -> None:
    """Bearer token from API_TOKEN (.env) on every API route except /health."""
    expected: str = request.app.state.settings.api_token
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if (
        not expected
        or scheme.lower() != "bearer"
        or not secrets.compare_digest(token.encode(), expected.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
