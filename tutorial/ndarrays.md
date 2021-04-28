# The N-dimensional array (ndarray) #

## Description ##

A ndarray is a fixed-size multi-dimensional container of items of the same type and size. The number of dimensions and items in an array is defined by its shape, which is a container of N non-negative integers that specify the sizes of each dimension. The type of items in the array is specified by a separate data-type object, one of which is associated with each ndarray.

Different ndarrays can share the same data, so that changes made in one ndarray may be visible in another. That is, a ndarray can be a "view" to another ndarray, and the data it is referring to is taken care of by the "base" ndarray.
[read more](https://numpy.org/doc/stable/reference/arrays.html)

## Pyccel ndarrays ##

Pyccel uses the same implementation as Numpy ndarrays with some rules due to the difference between the host language (Python) "dynamically typed / internal garbage collector" and the target languages such as C and Fortran which are statically typed languages and don't have a garbage collector.

Below we will show some rules that Pyccel has set to handles those differences.

### Dynamically and statically typed languages ###

Generally a variable in Pyccel should always keep its initial type, this also transfers to using the ndarrays.

#### incorrect example ####

```Python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
#(some code...)
a = np.array([1, 2, 3], dtype=int)
```

_OUTPUT_ :

```Shell
ERROR at annotation (semantic) stage
pyccel:
 |error [semantic]: ex.py [5]| Incompatible types in assignment (|a| real <-> int)
```

### Memory management ###

Pyccel makes a difference between ndarrays that own their data and the ones that don't.

Pyccel calls its own garbage collector when needed, but has a set of rules to do so:

-   Can not reassign ndarrays with different ranks.

    ```Python
    import numpy as np

    a = np.ones((10, 20))
    #(some code...)
    a = np.ones(10)
    ```

    _OUTPUT_ :

    ```Shell
    ERROR at annotation (semantic) stage
    pyccel:
     |error [semantic]: ex.py [4]| Incompatible redefinition (|a| real(10, 20) <-> real(10,))
    ```

This limitation is due to the fact that the rank of Fortran allocatable objects must be specified in their declaration.

-   Can not assign ndarrays that own their data one another.

    ```Python
    import numpy as np

    a = np.array([1, 2, 3, 4, 5])
    b = np.array([1, 2, 3])
    a = b
    ```

    _OUTPUT_ :

    ```Shell
    ERROR at annotation (semantic) stage
    pyccel:
     |error [semantic]: ex.py [5]| Arrays which own their data cannot become views on other arrays (a)
    ```

    This limitation is due to the fact that the ndarray **a** will have to go from a data owner to a pointer to the **b** ndarray data.

    _NOTE_: this limitation is not applied to assignments which reserve a new memory block.

    -   Python example:

        ```Python
        import numpy as np
        a = np.ones(20)
        #(some code...)
        a = np.ones(10)
        ```

    -   C equivalent:

        ```C
        #include "ndarrays.h"
        #include <stdlib.h>
        int main()
        {
            t_ndarray a;
            a = array_create(1, (int64_t[]){20}, nd_double);
            array_fill((double)1.0, a);
            /*(some code...)*/
            free_array(a);
            a = array_create(1, (int64_t[]){10}, nd_double);
            array_fill((double)1.0, a);
            free_array(a);
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_ex

        use, intrinsic :: ISO_C_BINDING

        implicit none

        real(C_DOUBLE), allocatable :: a(:)

        allocate(a(0:19_C_INT64_T))
        a = 1.0_C_DOUBLE
        !(some code...)
        if (any(size(a) /= [10_C_INT64_T])) then
        deallocate(a)
        allocate(a(0:9_C_INT64_T))
        end if
        a = 1.0_C_DOUBLE

        end program prog_ex
        ```

-   Can not reassign to a ndarray that has another pointer accessing its data.

    ```Python
    import numpy as np

    a = np.ones(10)
    b = a[:5]
    #(some code...)
    a = np.zeros(20)
    ```

    _OUTPUT_ :

    ```Shell
    ERROR at annotation (semantic) stage
    pyccel:
      |error [semantic]: ex.py [6]| Attempt to reallocate an array which is being used by another variable (a)
    ```

This limitation is set since we need to free the previous data when we reallocate the ndarray. In this case, this will cause the data pointed to by the view **b** to became inaccessible.

### Slicing and indexing ###

The indexing and slicing in Pyccel handles only the basic indexing of [numpy arrays](https://numpy.org/doc/stable/user/basics.indexing.html).

Some examples:

-   Python code:

    ```Python
    import numpy as np

    a = np.array([1, 3, 4, 5])
    a[0] = 0
    ```

    -   C equivalent:

        ```C
        #include <stdlib.h>
        #include "ndarrays.h"
        #include <stdint.h>
        int main()
        {
            t_ndarray a;
            a = array_create(1, (int64_t[]){4}, nd_int64);
            int64_t array_dummy_0001[] = {1, 3, 4, 5};
            memcpy(a.nd_int64, array_dummy_0001, a.buffer_size);
            a.nd_int64[get_index(a, 0)] = 0;
            free_array(a);
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_ex

        use, intrinsic :: ISO_C_BINDING

        implicit none

        integer(C_INT64_T), allocatable :: a(:)

        allocate(a(0:3_C_INT64_T))
        a = [1_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T, 5_C_INT64_T]
        a(0_C_INT64_T) = 0_C_INT64_T

        end program prog_ex
        ```

-   Python code:

    ```Python
    import numpy as np

    a = np.ones((10, 20))
    b = a[2:, :5]
    ```

    -   C equivalent:

        ```C
        #include "ndarrays.h"
        #include <stdlib.h>
        int main()
        {
            t_ndarray a;
            t_ndarray b;
            a = array_create(2, (int64_t[]){10, 20}, nd_double);
            array_fill((double)1.0, a);
            b = array_slicing(a, 2, new_slice(2, a.shape[0], 1), new_slice(0, 5, 1));
            free_array(a);
            free_pointer(b);
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_ex

        use, intrinsic :: ISO_C_BINDING

        implicit none

        real(C_DOUBLE), allocatable, target :: a(:,:)
        real(C_DOUBLE), pointer :: b(:,:)

        allocate(a(0:19_C_INT64_T, 0:9_C_INT64_T))
        a = 1.0_C_DOUBLE
        b(0:, 0:) => a(:4_C_INT64_T, 2_C_INT64_T:)

        end program prog_ex
        ```

-   Python code:

    ```Python
    import numpy as np

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = a[1]
    c = b[2]
    print(c)
    ```

    -   C equivalent:

        ```C
        #include <stdio.h>
        #include <stdint.h>
        #include <stdlib.h>
        #include "ndarrays.h"
        int main()
        {
            t_ndarray a;
            t_ndarray b;
            int64_t c;
            a = array_create(2, (int64_t[]){2, 4}, nd_int64);
            int64_t array_dummy_0001[] = {1, 2, 3, 4, 5, 6, 7, 8};
            memcpy(a.nd_int64, array_dummy_0001, a.buffer_size);
            b = array_slicing(a, 1, new_slice(1, 2, 1), new_slice(0, a.shape[1], 1));
            c = b.nd_int64[get_index(b, 2)];
            printf("%ld\n", c);
            free_array(a);
            free_pointer(b);
            return 0;
        }
        ```

    -   Fortran equivalent:

        ```Fortran
        program prog_ex

        use, intrinsic :: ISO_C_BINDING

        implicit none

        integer(C_INT64_T), allocatable, target :: a(:,:)
        integer(C_INT64_T), pointer :: b(:)
        integer(C_INT64_T) :: c

        allocate(a(0:3_C_INT64_T, 0:1_C_INT64_T))
        a = reshape([[1_C_INT64_T, 2_C_INT64_T, 3_C_INT64_T, 4_C_INT64_T], [ &
            5_C_INT64_T, 6_C_INT64_T, 7_C_INT64_T, 8_C_INT64_T]], [ &
            4_C_INT64_T, 2_C_INT64_T])
        b(0:) => a(:, 1_C_INT64_T)
        c = b(2_C_INT64_T)
        print *, c

        end program prog_ex
        ```

## Numpy [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) functions/properties progress in Pyccel ##

-   Supported [types](https://numpy.org/devdocs/user/basics.types.html):

    bool, int, int8, int16, int32, int64, float, float32, float64, complex64 and complex128. They can be used as cast functions too.

    Note: np.bool, np.int and np.float are just aliases to the Python native types, and are considered as a deprecated way to work with Python built-in types in NumPy.

-   Properties:

    -   real, imag, shape, amax, amin

-   Methods:

    -   sum