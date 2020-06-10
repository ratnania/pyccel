__all__ = [
        'slice_is_pointer_idx_0',
        'array_copy_is_pointer'
        ]

def slice_is_pointer_idx_0():
    from numpy import array, shape
    a = array([1, 2, 3, 4])
    b = a[1:]
    a[2] = 9
    return b[0], b[1], b[2], shape(b)[0]

# Issue 177
def array_copy_is_pointer():
    from numpy import array, shape
    a = array([1, 2, 3, 4])
    b = a
    a[2] = 9
    return b[0], b[1], b[2], b[3], shape(b)[0]
