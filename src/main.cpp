#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/LU>
#include <Eigen/SparseCore>

namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

Eigen::Ref<Eigen::MatrixXd> testOpenMPThreads(Eigen::Ref<Eigen::MatrixXd> matrix);

PYBIND11_MODULE(_functions, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: gaccord

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def(
        "testOpenMPThreads",
        &testOpenMPThreads,
        R"pbdoc(
            test function
        )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
