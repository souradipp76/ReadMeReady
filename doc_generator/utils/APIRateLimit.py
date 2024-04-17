class APIRateLimit:
    def __init__(self, max_concurrent_calls=50):
        self._queue = []
        self._in_progress = 0
        self._max_concurrent_calls = max_concurrent_calls

    def call_api(self, api_function):
        def execute_call():
            self._in_progress += 1
            try:
                result = api_function()
                return result
            except Exception as error:
                raise error  # Propagate the exception
            finally:
                self._in_progress -= 1
                self._dequeue_and_execute()

        self._queue.append(execute_call)

        # Trigger the dequeue and execute operation if under the limit
        if self._in_progress < self._max_concurrent_calls:
            self._dequeue_and_execute()

        # Since we're executing sequentially, there's no future to wait on.
        # Instead, return the result directly from the execute call.
        if self._queue:  # Check if there is anything left in the queue
            return self._queue.pop(0)()

    def _dequeue_and_execute(self):
        # Only dequeue if there are available slots and items in the queue
        while self._queue and self._in_progress < self._max_concurrent_calls:
            next_call = self._queue.pop(0) if self._queue else None
            if next_call:
                next_call()
