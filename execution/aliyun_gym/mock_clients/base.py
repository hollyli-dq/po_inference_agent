from Tea.exceptions import TeaException

class MockClientBase:
    """
    Base class for Mock Clients.
    Intercepts call_api and delegates to ActionRouter with product identification.
    """
    # Subclasses should override this
    PRODUCT_ID = "UNKNOWN"
    
    def __init__(self, state_store, chaos_injector, action_router):
        self._state_store = state_store
        self._chaos_injector = chaos_injector
        self._action_router = action_router

    def call_api(self, params, req, runtime):
        """
        Intercepts the API call and routes to the appropriate handler.
        Uses PRODUCT_ID to disambiguate APIs with the same name.
        """
        action = params.action
        query = req.query if req.query else {}
        
        if req.body:
            if isinstance(req.body, dict):
                query.update(req.body)
        
        # Dispatch to handler with product identification
        result = self._action_router.dispatch(action, query, product=self.PRODUCT_ID)
        
        # Check for error response
        if "Code" in result and "Message" in result:
            # It's an error response
            raise TeaException({
                "code": result["Code"],
                "message": result["Message"],
                "data": result
            })
            
        # Wrap result in a structure that TeaCore.from_map expects for Response object
        # The Response object usually has 'body', 'headers', 'statusCode'
        return {
            "body": result,
            "headers": {},
            "statusCode": 200
        }
