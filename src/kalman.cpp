#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

/*
Rauch–Tung–Striebel smoother (local-level: x_t = x_{t-1} + w_t, y_t = x_t + v_t)
Column-wise: input Y is (T, N) row-major; NaN means missing observation.
Q, R are positive scalars shared by all columns.
Returns an array (T, N) of smoothed states.
*/
static py::array_t<double> rts_smoother(py::array_t<double, py::array::c_style | py::array::forcecast> Y, double Q, double R)
{
    if (!(Q > 0.0) || !(R > 0.0))
    {
        throw std::invalid_argument("Q and R must be positive scalars.");
    }

    auto buf = Y.request();
    if (buf.ndim != 2)
    {
        throw std::invalid_argument("Input must be a 2D array of shape (T, N).");
    }

    const py::ssize_t T = buf.shape[0];
    const py::ssize_t N = buf.shape[1];
    if (T <= 0) {
        throw std::invalid_argument("T must be >= 1.");
    }

    const double* yptr = static_cast<const double*>(buf.ptr);

    // Output array with same shape as input
    auto out = py::array_t<double>(buf.shape);
    auto out_buf = out.request();

    double* x_s_ptr = static_cast<double*>(out_buf.ptr);
    
    // Numerics
    const double P0_diffuse_scale = 1e6;
    const double eps = 1e-12;

    // Iterate columns
    for (py::ssize_t n = 0; n < N; ++n)
    {
        // Per-series buffers
        std::vector<double> x_f(T), P_f(T), x_p(T), P_p(T), x_s(T), P_s(T);

        // Find first finite observation index t0
        py::ssize_t t0 = -1;
        for (py::ssize_t t = 0; t < T; ++t)
        {
            const double y = yptr[t * N + n];
            if (std::isfinite(y))
            {
                t0 = t;
                break;
            }
        }

        // If the whole column is missing, continue
        if (t0 < 0)
        {
            const double nan_value = std::numeric_limits<double>::quiet_NaN();
            for (py::ssize_t t = 0; t < T; ++t)
            {
                x_s_ptr[t * N + n] = nan_value;
            }

            continue;
        }

        // ---- Forward pass ----
        double x_prev = 0.0;
        double P_prev = P0_diffuse_scale * R;

        for (py::ssize_t t = 0; t <= t0; ++t)
        {
            const double x_pred = x_prev;
            const double P_pred = std::max(P_prev + Q, eps);

            x_p[t] = x_pred;
            P_p[t] = P_pred;

            if (t < t0)
            {
                // No update before t0
                x_f[t] = x_pred;
                P_f[t] = P_pred;
                x_prev = x_f[t];
                P_prev = P_f[t];
            }
            else
            {
                // Update with first finite observation
                const double y = yptr[t * N + n];
                const double v = y - x_pred;
                const double S = std::max(P_pred + R, eps);
                const double K = P_pred / S;
                const double x_filt = x_pred + K * v;

                // Joseph form
                const double one_minus_K = 1.0 - K;
                const double P_filt = std::max(one_minus_K * one_minus_K * P_pred + K * K * R, eps);

                x_f[t] = x_filt;
                P_f[t] = P_filt;

                x_prev = x_filt;
                P_prev = P_filt;
            }
        }

        // Standard Kalman filter with NaN handling
        for (py::ssize_t t = t0 + 1; t < T; ++t)
        {
            // Predict
            const double x_pred = x_prev;
            const double P_pred = std::max(P_prev + Q, eps);

            x_p[t] = x_pred;
            P_p[t] = P_pred;

            const double y = yptr[t * N + n];
            if (std::isfinite(y))
            {
                const double v = y - x_pred;
                const double S = std::max(P_pred + R, eps);
                const double K = P_pred / S;
                const double x_filt = x_pred + K * v;

                // Joseph form
                const double one_minus_K = 1.0 - K;
                const double P_filt = std::max(one_minus_K * one_minus_K * P_pred + K * K * R, eps);

                x_f[t] = x_filt;
                P_f[t] = P_filt;

                x_prev = x_filt;
                P_prev = P_filt;
            }
            else
            {
                // Missing obs
                x_f[t] = x_pred;
                P_f[t] = P_pred;

                x_prev = x_f[t];
                P_prev = P_f[t];
            }
        }

        // ---- Backward pass ----
        x_s[T - 1] = x_f[T - 1];
        P_s[T - 1] = P_f[T - 1];

        for (py::ssize_t t = T - 2; t >= 0; --t)
        {
            const double P_t1_t = std::max(P_p[t + 1], eps);
            const double J = P_f[t] / P_t1_t;
            const double x_smooth = x_f[t] + J * (x_s[t + 1] - x_p[t + 1]);
            const double P_smooth = std::max(P_f[t] + J * J * (P_s[t + 1] - P_t1_t), eps);

            x_s[t] = x_smooth;
            P_s[t] = P_smooth;

            // avoid signed underflow on some compilers
            if (t == 0)
            { 
                break; 
            }
        }

        // Write column back to output
        for (py::ssize_t t = 0; t < T; ++t)
        {
            x_s_ptr[t * N + n] = x_s[t];
        }
    }

    return out;
}

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Column-wise RTS Kalman smoother (local-level) for (T, N) arrays; NaNs treated as missing.";
    m.def("rts_smoother", &rts_smoother, py::arg("Y"), py::arg("Q"), py::arg("R"), "Return smoothed states (T, N) for a local-level model with shared Q, R.");
}