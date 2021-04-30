import aiohttp
import asyncio
import uproot

class AIOHTTPSource(uproot.source.chunk.Source):
    "Experimental data source for uproot that uses asyncio and connection pooling"

    def __init__(self, file_path, ssl_context=None, tcp_connection_limit=10, **options):
        self._file_path = file_path
        self._ssl_context = ssl_context
        self._tcp_connection_limit = tcp_connection_limit

        async def create_session():
            conn = aiohttp.TCPConnector(limit=self._tcp_connection_limit)
            return aiohttp.ClientSession(connector=conn)

        self._loop = asyncio.get_event_loop()
        self._session = self._loop.run_until_complete(create_session())

    @property
    def closed(self):
        return self._session.closed

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._loop.run_until_complete(self._session.close())

    async def get(self, start, stop, notifications=None):
        async with self._session.get(
            self._file_path,
            headers={"Range": f"bytes={start}-{stop - 1}"},
            ssl=self._ssl_context,
        ) as resp:
            content = await resp.read()
            future = uproot.source.futures.TrivialFuture(content)
            chunk = uproot.source.chunk.Chunk(self, start, stop, future)
            if notifications is not None:
                notifications.put(chunk)
            return chunk

    def chunk(self, start, stop):
        return self._loop.run_until_complete(self.get(start, stop))

    def chunks(self, ranges, notifications):

        async def achunks():
            return await asyncio.gather(
                *[self.get(start, stop, notifications) for start, stop in ranges]
            )

        return self._loop.run_until_complete(achunks())
