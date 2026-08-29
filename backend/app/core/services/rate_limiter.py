from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Callable

def resilient_get_remote_address(request=None) -> str:
    """
    Resilient key function for rate limiter that handles wrapped/transformed request objects.
    Falls back to generic 'user' if request is not available or not a proper Starlette Request.
    This prevents slowapi type checking errors with middleware-wrapped requests.
    """
    if request is None:
        return "user"
    
    # Try to get the client IP from the request object
    try:
        # Standard Starlette Request
        return get_remote_address(request)
    except (AttributeError, TypeError):
        pass
    
    # Fallback for wrapped or non-standard request objects
    try:
        if hasattr(request, 'client') and request.client:
            return request.client[0]
    except:
        pass
    
    try:
        if hasattr(request, 'headers'):
            x_forwarded_for = request.headers.get('x-forwarded-for')
            if x_forwarded_for:
                return x_forwarded_for.split(',')[0].strip()
    except:
        pass
    
    # Last resort: generic fallback
    return "user"

limiter = Limiter(key_func=resilient_get_remote_address)
