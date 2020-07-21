import pyxrootd.client

class XRDFile(pyxrootd.client.File):
    "Implement python file interface for XRootD (as far as ParquetFile needs it)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos = 0
        self._size = None
        # TODO: manage this properly
        self.closed = False

    @property
    def size(self):
        if self._size is None:
            self._size = self.stat()[1]["size"]
        return self._size

    def seek(self, pos, whence=0):
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos -= pos
        elif whence == 2:
            self._pos = self.size - pos
        else:
            raise ValueError("Whence has to be 0, 1 or 2")

    def tell(self):
        return self._pos

    def read(self, size=-1):
        if size < 0:
            size = self.size - self._pos
        status, bytes = super().read(offset=self._pos, size=size)
        if status["status"] != 0:
            raise IOError(status)
        return bytes

if __name__ == "__main__":

    import pyarrow.parquet as pq
    f = XRDFile()
    f.open("root://lcg-lrz-rootd.grid.lrz.de:1094/pnfs/lrz-muenchen.de/data/atlas/dq2/atlaslocalgroupdisk/rucio/user/nihartma/c4/40/testdata_physlite.parquet")

    #pqf = pq.ParquetFile(f)

    import awkward1 as ak
    ar = ak.from_parquet(f, columns=["AnalysisElectronsAuxDyn.pt"])
