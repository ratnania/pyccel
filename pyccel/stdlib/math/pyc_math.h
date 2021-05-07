/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef         PYC_MATH_H
#define         PYC_MATH_H

#include <stdint.h>

#define MOD_PYC1(N, M) (((N % M) + M) % M)
#define FMOD_PYC1(N, M) (fmod(fmod(N, M) + M, M))

#define MOD_PYC2(N, M) (N < 0 ? (N % M) + M : (N % M))
#define FMOD_PYC2(N, M) (N < 0 ? fmod(N, M) + M : fmod(N, M))

#define MOD_PYC3(N, M) ((N % M) >= 0 ? N % M : (N % M) + M)
#define FMOD_PYC3(N, M) (fmod(N, M) >= 0 ? fmod(N, M) : fmod(N, M) + M)

int64_t         pyc_factorial(int64_t n);
int64_t         pyc_gcd (int64_t a, int64_t b);
int64_t         pyc_lcm (int64_t a, int64_t b);
double          pyc_radians(double degrees);
double          pyc_degrees(double radians);

#endif
