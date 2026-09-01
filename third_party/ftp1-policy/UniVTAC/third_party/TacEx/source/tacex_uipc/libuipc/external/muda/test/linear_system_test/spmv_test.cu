#include <catch2/catch.hpp>
#include <muda/muda.h>
#include <muda/ext/linear_system.h>
using namespace muda;
using namespace Eigen;

//using T                = float;
//constexpr int BlockDim = 3;
template <typename T, int BlockDimM, int BlockDimN>
void test_sparse_matrix(int row_size, int col_size, int non_zero_count)
{
    int dimensionM = BlockDimM * row_size;
    int dimensionN = BlockDimN * col_size;


    LinearSystemContext ctx;

    Eigen::MatrixX<T> dense_A = Eigen::MatrixX<T>::Zero(dimensionM, dimensionN);
    Eigen::VectorX<T> dense_x = Eigen::VectorX<T>::Zero(dimensionN);
    dense_x.setRandom();
    Eigen::MatrixX<T> host_A;

    // setup device vector
    DeviceDenseVector<T> x = dense_x;
    DeviceDenseVector<T> b(dimensionM);

    std::vector<int> row_indices(non_zero_count);
    std::vector<int> col_indices(non_zero_count);
    std::vector<Eigen::Matrix<T, BlockDimM, BlockDimN>> blocks(non_zero_count);

    for(int i = 0; i < non_zero_count; ++i)  // create random blocks
    {
        Eigen::Vector2f index =
            (Eigen::Vector2f::Random() + Eigen::Vector2f::Ones()) / 2.0f;
        row_indices[i] = index.x() * row_size;
        col_indices[i] = index.y() * col_size;
        if(row_indices[i] == row_size)
            row_indices[i] = row_size - 1;
        if(col_indices[i] == col_size)
            col_indices[i] = col_size - 1;
        blocks[i] = Eigen::Matrix<T, BlockDimM, BlockDimN>::Random();
    }

    for(int i = 0; i < non_zero_count; ++i)  // set dense matrix
    {
        Eigen::Block<Eigen::MatrixX<T>, -1, -1> block = dense_A.block(
            row_indices[i] * BlockDimM, col_indices[i] * BlockDimN, BlockDimM, BlockDimN);

        block += blocks[i];
    }

    Eigen::VectorX<T> ground_truth = dense_A * dense_x;
    Eigen::VectorX<T> host_b;

    DeviceTripletMatrix<T, BlockDimM, BlockDimN> A_triplet;
    A_triplet.reshape(row_size, row_size);
    A_triplet.resize_triplets(non_zero_count);

    A_triplet.row_indices().copy_from(row_indices.data());
    A_triplet.col_indices().copy_from(col_indices.data());
    A_triplet.values().copy_from(blocks.data());
    {
        ctx.spmv(A_triplet.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }

    DeviceBCOOMatrix<T, BlockDimM, BlockDimN> A_bcoo;
    ctx.convert(A_triplet, A_bcoo);
    {
        b.fill(0);
        ctx.spmv(A_bcoo.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));
    }

    DeviceDenseMatrix<T> A;
    DeviceCOOMatrix<T>   A_coo;
    ctx.convert(A_bcoo, A_coo);
    A.fill(0);
    ctx.convert(A_coo, A);
    {
        b.fill(0);
        ctx.spmv(A_coo.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));

        A.copy_to(host_A);
        REQUIRE(host_A.isApprox(dense_A));
    }


    ctx.convert(A_bcoo, A);
    {
        b.fill(0);
        ctx.mv(A.cview(), x.cview(), b.view());
        ctx.sync();
        b.copy_to(host_b);
        REQUIRE(host_b.isApprox(ground_truth));

        A.copy_to(host_A);
        REQUIRE(host_A.isApprox(dense_A));
    }

    if constexpr(BlockDimM == BlockDimN)
    {
        DeviceBSRMatrix<T, BlockDimM> A_bsr;
        ctx.convert(A_bcoo, A_bsr);
        {
            b.fill(0);
            ctx.spmv(A_bsr.cview(), x.cview(), b.view());
            ctx.sync();
            b.copy_to(host_b);
            REQUIRE(host_b.isApprox(ground_truth));
        }

        DeviceCSRMatrix<T> A_csr;
        ctx.convert(A_bsr, A_csr);
        {
            b.fill(0);
            ctx.spmv(A_csr.cview(), x.cview(), b.view());
            ctx.sync();
            b.copy_to(host_b);
            REQUIRE(host_b.isApprox(ground_truth));
        }

        A_csr.clear();
        ctx.convert(A_coo, A_csr);
        {
            b.fill(0);
            ctx.spmv(A_csr.cview(), x.cview(), b.view());
            ctx.sync();
            b.copy_to(host_b);
            REQUIRE(host_b.isApprox(ground_truth));
        }
    }
}

TEST_CASE("spmv", "[linear_system]")
{
    test_sparse_matrix<float, 3, 3>(10, 10, 40);
    test_sparse_matrix<float, 3, 3>(100, 100, 400);
    test_sparse_matrix<float, 3, 3>(1000, 1000, 4000);

    test_sparse_matrix<float, 12, 12>(10, 10, 24);
    test_sparse_matrix<float, 12, 12>(100, 100, 888);
    test_sparse_matrix<float, 12, 12>(1000, 1000, 7992);

    test_sparse_matrix<float, 3, 1>(10, 10, 40);
    test_sparse_matrix<float, 3, 1>(100, 100, 400);
    test_sparse_matrix<float, 3, 1>(1000, 1000, 4000);


    test_sparse_matrix<float, 1, 3>(10, 10, 40);
    test_sparse_matrix<float, 1, 3>(100, 100, 400);
    test_sparse_matrix<float, 1, 3>(1000, 1000, 4000);


    test_sparse_matrix<float, 3, 2>(10, 10, 40);
    test_sparse_matrix<float, 3, 2>(100, 100, 400);
    test_sparse_matrix<float, 3, 2>(1000, 1000, 4000);


    test_sparse_matrix<float, 2, 3>(10, 10, 40);
    test_sparse_matrix<float, 2, 3>(100, 100, 400);
    test_sparse_matrix<float, 2, 3>(1000, 1000, 4000);

    test_sparse_matrix<float, 12, 1>(10, 10, 24);
    test_sparse_matrix<float, 12, 1>(100, 100, 637);
    test_sparse_matrix<float, 12, 1>(1000, 1000, 7456);

    test_sparse_matrix<float, 1, 12>(10, 10, 24);
    test_sparse_matrix<float, 1, 12>(100, 100, 537);
    test_sparse_matrix<float, 1, 12>(1000, 1000, 7456);

    test_sparse_matrix<float, 12, 3>(10, 10, 24);
    test_sparse_matrix<float, 12, 3>(100, 100, 637);
    test_sparse_matrix<float, 12, 3>(1000, 1000, 7456);

    test_sparse_matrix<float, 3, 12>(10, 10, 24);
    test_sparse_matrix<float, 3, 12>(100, 100, 637);
    test_sparse_matrix<float, 3, 12>(1000, 1000, 7456);
}