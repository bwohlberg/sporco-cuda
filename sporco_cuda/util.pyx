# cython: embedsignature=True

cdef extern int get_device_count()
cdef extern int get_current_device(int *dev)
cdef extern int set_current_device(int dev)
cdef extern int get_memory_info(size_t* free, size_t* total)
cdef extern char* get_device_name(int dev)


def device_count():

    return get_device_count()


def current_device(newdev=None):

    cdef int dev
    if newdev is not None:
        dev = newdev
        err = set_current_device(dev)
        if err != 0:
            raise RuntimeError('CUDA error %d while setting device' % err)
    err = get_current_device(&dev)
    if err != 0:
        raise RuntimeError('CUDA error %d while getting current device' % err)
    return dev


def memory_info():

    cdef size_t free, total
    err = get_memory_info(&free, &total)
    if err != 0:
        raise RuntimeError('CUDA error %d' % err)
    return free, total


def device_name(int dev=0):

    cdef char* cstr
    cstr = get_device_name(dev)
    if cstr == NULL:
        return None
    else:
        return (<bytes> cstr).decode('UTF-8')
