class DataFile:
    def open(cls, path, mode):
        """
        Open a data file.  The base implementation of this function just
        opens and returns a standard python file object.
        Subclasses can override to either open files using different openers
        (like fitsio.FITS), or, for more specific data types, return an
        instance of the class itself to use as an intermediary for the file.
        """
        return open(path, mode)

    @classmethod
    def make_name(cls, tag):
        if cls.suffix:
            return f'{tag}.{cls.suffix}'
        else:
            return tag


class DummyFile(DataFile):
    suffix = ''


class DirFile(DataFile):
    suffix = None


class FitsFile(DataFile):
    """
    A data file in the FITS format.
    Using these files requires the fitsio package.
    """
    suffix = 'fits'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import fitsio
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        if mode == 'w':
            mode = 'rw'
        return fitsio.FITS(path, mode=mode, **kwargs)


class ASCIIFile(DataFile):
    """
    A data file in human-readable ASCII.
    """
    suffix = 'txt'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import fitsio
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        return open(path, mode)


class BinaryFile(DataFile):
    """
    A binary data file
    """
    suffix = 'dat'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import fitsio
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        return open(path, mode)


class NpzFile(DataFile):
    """
    A binary data file
    """
    suffix = 'npz'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import fitsio
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        return open(path, mode)



class NpyFile(DataFile):
    """
    A .npy data file
    """
    suffix = 'npy'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import numpy as np
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        return np.load(path)

class SACCFile(DataFile):
    """
    A SACC file
    """
    suffix = 'sacc'

    @classmethod
    def open(cls, path, mode, **kwargs):
        import fitsio
        # Fitsio doesn't have pure 'w' modes, just 'rw'.
        # Maybe we should check if the file already exists here?
        return open(path, mode)
