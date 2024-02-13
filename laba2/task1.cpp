#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include <cmath>
#include "vectors_and_matrices/array_types.hpp"

#include <omp.h>

using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;
const double PI = 3.141592653589793;

template <class T>
void fill_random_sin(vec<T> x, T ampl_max, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(-ampl_max, ampl_max);

    ptrdiff_t n = x.length();
    for (int imode=1; imode<=10; imode++)
    {
        T ampl = dist(rng);
        for (ptrdiff_t i = 0; i < n; i++)
        {
            x(i) += ampl * sin(imode * (PI * (i+0.5)) / n);
        }
    }
}

double xAx(vec<double> x)
{
    // Вычисляет x' * A * x,
    // где A - оператор дискретного приближения 2 производной
    double prod = 0;
    ptrdiff_t n = x.length();
    ptrdiff_t i;
    vec<double> tmp(n);
    prod += x(0) * (-3 * x(0) + x(1));
    prod += x(n-1) * (x(n-2) - 3 * x(n-1));

    #pragma omp parallel for
    for (i = 1; i < n-1; i++)
    {
        tmp(i) = x(i) * (x(i-1) - 2 * x(i) + x(i+1));
    }

    #pragma omp parallel for reduction (+:prod)
    for (i=1; i<n-1; i++)
    {
        prod += tmp(i);
    }

    return prod;
}

// res <- rhp - alpha * A x
// A - finite-difference operator
void update_residual(vec<double> res, vec<double> x, vec<double> rhp, double alpha)
{
    ptrdiff_t n = res.length();
    res(0) = rhp(0) - alpha * (-3 * x(0) + x(1));
    res(n-1) = rhp(n-1) - alpha * (x(n-2) - 3 * x(n-1));
    ptrdiff_t i;

    #pragma omp parallel for
    for (i=1; i < n-1; i++)
    {
        res(i) = rhp(i) - alpha * (x(i-1) - 2*x(i) + x(i+1));
    }
}

template <class T>
T dot(vec<T> a, vec<T> b)
{
    T sum(0);
    ptrdiff_t i;
    ptrdiff_t n = a.length();
    vec<T> tmp(n);

    #pragma omp parallel for
    for (i = 0; i < n; i++)
    {
        tmp(i) = a(i) * b(i);
    }

    #pragma omp parallel for reduction (+:sum)
    for (i=0; i<n; i++)
    {
        sum += tmp(i);
    }
    return sum;
}

void solve_poisson(vec<double> u, vec<double> f, double u0, double uL, double h)
{
    double factor;
    ptrdiff_t n = f.length();
    vec<double> rhp(n);
    vec<double> residual(n);
    vec<double> dir(n);

    ptrdiff_t i, j;
    #pragma omp parallel for
    for (i = 0; i < n; i++)
    {
        u(i) = u0 + (uL - u0) * (i + 0.5) / n;
        rhp(i) = h * h * f(i);
    }


    rhp(0) -= 2 * u0;
    rhp(n-1) -= 2 * uL;

    update_residual(residual, u, rhp, 1.0);

    #pragma omp parallel for
    for (i = 0; i < n; i++)
    {
        dir(i) = residual(i);
    }

    double rkrk = dot(residual, residual);
    double alpha, d, beta;

    for (i = 0; i < n; i++)
    {
        alpha = rkrk / xAx(dir);

        // x <- x + alpha * dir
        #pragma omp parallel for
        for (j = 0; j < n; j++)
        {
            u(j) += alpha * dir(j);
        }

        // r <- r + alpha * A * dir
        update_residual(residual, dir, residual, alpha);

        d = dot(residual, residual);
        beta = d / rkrk;
        rkrk = d;

        // dir <- r + beta * dir
        #pragma omp parallel for
        for (j = 0; j < n; j++)
        {
            dir(j) = residual(j) + beta * dir(j);
        }
    }
}

int main(int argc, char* argv[])
{
    ptrdiff_t n;

    std::cin >> n;
    vec<double> u(n), f(n);

    fill_random_sin(f, 5.0, 9876);

    double L = PI;

    double dx = L / n;

    double t0 = omp_get_wtime();

    // solve Poisson equation with u_0 = u_L = 0
    solve_poisson(u, f, 0.0, 0.0, dx);

    double t1 = omp_get_wtime();

    std::cout << std::endl;
    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
              << "Timing: " << t1 - t0 << " sec\n"
              << "u[n/2] = " << u(n/2)
              << std::endl;
    return 0;
}
