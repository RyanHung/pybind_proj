#define EIGEN_USE_BLAS

extern "C" {
    #include <cblas.h>  // CBLAS header for standard BLAS
}

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>
#include <omp.h>

using namespace Eigen;
using namespace std;

double sgn(double val) {
    return (double(0) < val) - (val < double(0));
}

void sthreshmat(MatrixXd & x,
                double tau,
                const MatrixXd & t){

    MatrixXd tmp1(x.cols(), x.cols());
    MatrixXd tmp2(x.cols(), x.cols());

    tmp1 = x.array().unaryExpr(ptr_fun(sgn));
    tmp2 = (x.cwiseAbs() - tau*t).cwiseMax(0.0);

    x = tmp1.cwiseProduct(tmp2);

    return;
}

Eigen::Ref<Eigen::MatrixXd> testOpenMPThreads(Eigen::Ref<Eigen::MatrixXd> matrix) {
    int num_threads = omp_get_max_threads(); // Set the number of threads
    omp_set_num_threads(num_threads);

    // Initialize a 4x4 matrix with random values
    matrix = Eigen::MatrixXd::Random(4, 4);

    // Perform parallel operations using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            // Simple operation: multiply each element by 2
            matrix(i, j) = matrix(i, j) * 2;
        }
    }

    return matrix;
}

Eigen::Ref<Eigen::MatrixXd> testBLAS(Eigen::Ref<Eigen::MatrixXd> matrix)
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(matrix.rows(), matrix.cols());
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(matrix.rows(), matrix.cols());

    double alpha = 1.0;  // Scaling factor for A * B
    double beta = 0.0;   // Scaling factor for C

    // Call CBLAS dgemm (double-precision general matrix-matrix multiplication)
    cblas_dgemm(CblasRowMajor,  // Indicate row-major storage
                CblasNoTrans,    // A is not transposed
                CblasNoTrans,    // B is not transposed
                matrix.rows(), matrix.cols(), matrix.cols(), // Matrix dimensions
                alpha,            // alpha * A * B
                A.data(), A.cols(),    // Pointer to A, leading dimension of A
                B.data(), B.cols(),    // Pointer to B, leading dimension of B
                beta,             // beta * C
                matrix.data(), matrix.cols());   // Pointer to C, leading dimension of C

    return matrix;
}