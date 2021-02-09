/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef NDARRAYS_H
# define NDARRAYS_H

# include <stdlib.h>
# include <complex.h>
# include <string.h>
# include <stdio.h>
# include <stdarg.h>
# include <stdbool.h>
# include <stdint.h>

/* mapping the function array_fill to the correct type */
# define array_fill(c, arr) _Generic((c), int64_t : _array_fill_int64,\
                                        int32_t : _array_fill_int32,\
                                        int16_t : _array_fill_int16,\
                                        int8_t : _array_fill_int8,\
                                        float : _array_fill_float,\
                                        double : _array_fill_double,\
                                        bool : _array_fill_bool,\
                                        float complex : _array_fill_cfloat,\
                                        double complex : _array_fill_cdouble)(c, arr)

typedef struct  s_slice
{
    int32_t start;
    int32_t end;
    int32_t step;
}               t_slice;

/*
** Map e_types enum to numpy NPY_TYPES enum
** ref: numpy_repo: numpy/numpy/core/include/numpy/ndarraytypes.h
*/
enum e_types
{
        nd_bool     = 0,
        nd_int8     = 1,
        nd_int16    = 3,
        nd_int32    = 5,
        nd_int64    = 6,
        nd_float    = 7,
        nd_double   = 9,
        nd_cfloat   = 11,
        nd_cdouble  = 13
};

typedef struct  s_ndarray
{
    /* raw data buffer*/
    union {
            char            *raw_data;
            int8_t          *nd_int8;
            int16_t         *nd_int16;
            int32_t         *nd_int32;
            int64_t         *nd_int64;
            float           *nd_float;
            double          *nd_double;
            bool            *nd_bool;
            double complex  *nd_cdouble;
            float  complex  *nd_cfloat;
            };
    /* number of dimensions */
    int32_t                 nd;
    /* shape 'size of each dimension' */
    int64_t                 *shape;
    /* strides 'number of bytes to skip to get the next element' */
    int64_t                 *strides;
    /* type of the array elements */
    enum e_types            type;
    /* type size of the array elements */
    int32_t                 type_size;
    /* number of element in the array */
    int32_t                 length;
    /* size of the array */
    int32_t                 buffer_size;
    /* True if the array does not own the data */
    bool                    is_view;
}               t_ndarray;

/* functions prototypes */

/* allocations */
void        stack_array_init(t_ndarray *arr);
t_ndarray   array_create(int32_t nd, int64_t *shape, enum e_types type);
void        _array_fill_int8(int8_t c, t_ndarray arr);
void        _array_fill_int16(int16_t c, t_ndarray arr);
void        _array_fill_int32(int32_t c, t_ndarray arr);
void        _array_fill_int64(int64_t c, t_ndarray arr);
void        _array_fill_float(float c, t_ndarray arr);
void        _array_fill_double(double c, t_ndarray arr);
void        _array_fill_bool(bool c, t_ndarray arr);
void        _array_fill_cfloat(float complex c, t_ndarray arr);
void        _array_fill_cdouble(double complex c, t_ndarray arr);

/* slicing */
                /* creating a Slice object */
t_slice     new_slice(int32_t start, int32_t end, int32_t step);
                /* creating an array view */
t_ndarray   array_slicing(t_ndarray arr, int n, ...);

/* assigns */
void        alias_assign(t_ndarray *dest, t_ndarray src);

/* free */
int32_t         free_array(t_ndarray dump);
int32_t         free_pointer(t_ndarray dump);

/* indexing */
int64_t         get_index(t_ndarray arr, ...);

/* data converting between numpy and ndarray */
int64_t     *numpy_to_ndarray_strides(int64_t *np_strides, int type_size, int nd);
int64_t     *numpy_to_ndarray_shape(int64_t *np_shape, int nd);

#endif
