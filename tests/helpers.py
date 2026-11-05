"""Small helpers shared by several test files (conftest.py puts tests/ on the path)."""


class SerialPool:
    """Stands in for multiprocessing.Pool: runs every job in this process, in order,
    so tests stay small and their results are the same as with a real pool."""

    def map(self, fn, jobs, chunksize=None):
        return [fn(j) for j in jobs]

    def imap(self, fn, jobs, chunksize=1):
        return (fn(j) for j in jobs)
