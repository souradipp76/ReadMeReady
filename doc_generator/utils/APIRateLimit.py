class APIRateLimit:
    def __init__(self, max_concurrent_calls=50):
        self._queue = []
        self._in_progress = 0
        self._max_concurrent_calls = max_concurrent_calls

    def call_api(self, api_function):
        # Create a promise-like event using a Future
        future = asyncio.Future()

        def execute_call():
            nonlocal future
            self._in_progress += 1
            try:
                result = api_function()
                if not future.done():
                    future.set_result(result)
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                self._in_progress -= 1
                self._dequeue_and_execute()

        self._queue.append(execute_call)

        # Trigger the dequeue and execute operation when there are available slots for concurrent calls
        if self._in_progress < self._max_concurrent_calls:
            self._dequeue_and_execute()

        return future

    def _dequeue_and_execute(self):
        while self._queue and self._in_progress < self._max_concurrent_calls:
            next_call = self._queue.pop(0) if self._queue else None
            if next_call:
                next_call()
