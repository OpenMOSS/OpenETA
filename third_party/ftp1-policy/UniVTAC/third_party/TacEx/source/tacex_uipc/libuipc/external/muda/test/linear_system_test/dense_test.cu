#include <catch2/catch.hpp>
#include <muda/muda.h>

#include <muda/ext/eigen/eigen_dense_cxx20.h>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;

template <typename T>
void test_dense_vector(int dim)
{
    LinearSystemContext ctx;
    VectorX<T>          h_x   = VectorX<T>::Ones(dim);
    VectorX<T>          h_y   = VectorX<T>::Ones(dim);
    VectorX<T>          h_res = VectorX<T>::Zero(dim);

    DeviceDenseVector<T> x = h_x;
    DeviceDenseVector<T> y = h_y;
    DeviceDenseVector<T> z = h_res;

    // dot
    T res_dot = ctx.dot(x.cview(), y.cview());
    T gt_dot  = h_x.dot(h_y);

    REQUIRE(Approx(res_dot) == gt_dot);

    // norm
    T res_norm = ctx.norm(x.cview());
    T gt_norm  = h_x.norm();

    REQUIRE(Approx(res_norm) == gt_norm);

    // axpby
    T alpha = 2.0;
    T beta  = 3.0;
    ctx.axpby(alpha, x.cview(), beta, y.view());
    y.copy_to(h_res);
    VectorX<T> gt_axpby = alpha * h_x + beta * h_y;
    REQUIRE(h_res.isApprox(gt_axpby));

    // plus
    y = h_y;
    ctx.plus(x.cview(), y.cview(), z.view());
    z.copy_to(h_res);
    VectorX<T> gt_plus = h_x + h_y;
    REQUIRE(h_res.isApprox(gt_plus));
}

void test_dense_vector_view()
{
    LinearSystemContext ctx;
    VectorX<double>     h_x = VectorX<double>::Ones(100);
    VectorX<double>     h_y = VectorX<double>::Ones(100);
    VectorX<double>     h_res(100);

    DeviceDenseVector<double> x = h_x;
    DeviceDenseVector<double> y = h_y;

    DenseVectorView<double> x_view = x.view();

    // non-const to const conversion
    CDenseVectorView<double> x_cview = x_view;

    Launch()
        .file_line(__FILE__, __LINE__)
        .apply(
            [x = x_view.viewer().name("x"), y = y.viewer().name("y")] __device__() {

            });
}

TEST_CASE("dense_vector", "[linear_system]")
{
    test_dense_vector<double>(1000);
    test_dense_vector<float>(1000);
    test_dense_vector_view();
}


void test_dense_matrix_view()
{
    LinearSystemContext ctx;
    MatrixXd            h_A = MatrixXd::Ones(100, 100);
    MatrixXd            h_B = MatrixXd::Ones(100, 100);
    MatrixXd            h_res(100, 100);

    DeviceDenseMatrix<double> A = h_A;
    DeviceDenseMatrix<double> B = h_B;

    DenseMatrixView<double> A_view = A.view();

    // non-const to const conversion
    CDenseMatrixView<double> A_cview = A_view;

    Launch()
        .file_line(__FILE__, __LINE__)
        .apply(
            [A = A_view.viewer().name("A"), B = B.viewer().name("B")] __device__() {

            });
}

TEST_CASE("dense_matrix", "[linear_system]")
{
    test_dense_matrix_view();
}
