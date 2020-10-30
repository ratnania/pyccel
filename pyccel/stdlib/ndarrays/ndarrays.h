#ifndef NDARRAYS_H
# define NDARRAYS_H

# include <stdlib.h>
# include <complex.h>
# include <string.h>
# include <stdio.h>
# include <stdarg.h>
# include <stdbool.h> 

typedef struct  s_slice_data{
    int start;
    int end;
    int step;
}               t_slice;

enum e_types{
        nd_int,
        nd_float,
        nd_double,
        nd_complex_double
            };

typedef struct  s_ndarray
{
    /* raw data buffer*/
    union {
            char            *raw_data;
            int             *int_nd;
            float           *float_nd;
            double          *double_nd;
            double complex  *complex_double;
            };
    /* number of dimmensions */
    int             nd;
    /* shape 'size of each dimmension' */
    int             *shape;
    /* strides 'number of bytes to skip to get the next element' */
    int             *strides;
    /* type of the array elements */
    enum e_types    type; // TODO : make it into an enum
    int             type_size; // TODO : make it into an enum
    int             length;
    bool            is_slice;
}               t_ndarray;

/* functions prototypes */

/* allocations */
// t_ndarray   *array_create(char *temp, int nd, int *shape, int type);
t_ndarray   array_init(char *temp, int nd, int *shape, enum e_types type, int type_size);
// t_ndarray   *array_ones(int nd, int *shape, int type);
// t_ndarray   *array_zeros(int nd, int *shape, int type);
// t_ndarray   *array_full(char *temp, int nd, int *shape, int type);

/* slicing */
t_slice     slice_data(int start, int end, int step);
t_ndarray   slice_make(t_ndarray p, ...);

/* free */
int         free_array(t_ndarray dump);

/* indexing */
int         get_index(t_ndarray arr, ...);

#endif
