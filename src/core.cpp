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