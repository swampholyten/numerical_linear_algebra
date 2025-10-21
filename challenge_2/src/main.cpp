#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <iostream>
#include <lis.h>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

class ChallengeTwo {
private:
  Eigen::MatrixXd Ag; // Adjacency matrix for small graph
  Eigen::VectorXd vg; // Degree vector for small graph
  Eigen::MatrixXd Dg; // Degree diagonal matrix for small graph
  Eigen::MatrixXd Lg; // Laplasian for small graph

  Eigen::SparseMatrix<double> As; // Adjacency matrix for social network
  Eigen::VectorXd vs;             // Degree vector for social network
  Eigen::SparseMatrix<double> Ds; // Degree diagonal matrix for social network
  Eigen::SparseMatrix<double> Ls; // Laplasian for social network

public:
  void create_small_graph() {
    Ag = Eigen::MatrixXd::Zero(9, 9);

    Ag(0, 1) = Ag(0, 3) = 1;

    Ag(1, 0) = Ag(1, 2) = 1;

    Ag(2, 1) = Ag(2, 3) = Ag(2, 4) = 1;

    Ag(3, 0) = Ag(3, 2) = 1;

    Ag(4, 2) = Ag(4, 5) = Ag(4, 7) = Ag(4, 8) = 1;

    Ag(5, 4) = Ag(5, 6) = 1;

    Ag(6, 5) = Ag(6, 7) = Ag(6, 8) = 1;

    Ag(7, 4) = Ag(7, 6) = Ag(7, 8) = 1;

    Ag(8, 4) = Ag(8, 6) = Ag(8, 7) = 1;

    double frobenius_norm = Ag.norm();
    std::cout << "Frobenius norm of Ag is: " << frobenius_norm << std::endl;
  }

  void create_graph_laplasian() {
    vg = Ag.rowwise().sum();

    Dg = Eigen::MatrixXd::Zero(9, 9);

    for (int i = 0; i < 9; i++) {
      Dg(i, i) = vg(i);
    }

    // L = D - A
    Lg = Dg - Ag;

    Eigen::VectorXd x = Eigen::VectorXd::Ones(9);
    Eigen::VectorXd y = Lg * x;

    double euclidean_norm = y.norm();

    bool is_symmetric = (Lg - Lg.transpose()).norm() < 1e-10;

    std::cout << "Is Lg symmetric? " << (is_symmetric ? "Yes" : "No")
              << std::endl;
  }

  void find_eigenvalues() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Lg);

    if (solver.info() != Eigen::Success) {
      std::cerr << "Eigen value computation failed" << std::endl;
      return;
    }

    Eigen::VectorXd eigenvalues = solver.eigenvalues();

    std::cout << "Smallest eigenvalue: " << eigenvalues.minCoeff() << std::endl;
    std::cout << "Biggest eigenvalue: " << eigenvalues.maxCoeff() << std::endl;
  }

  void find_fielder_vector() {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(Lg);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    double second_smallest = 1e10;
    int second_smallest_idx = -1;

    for (int i = 0; i < eigenvalues.size(); i++) {
      if (eigenvalues(i) > 1e-10 && eigenvalues(i) < second_smallest) {
        second_smallest = eigenvalues(i);
        second_smallest_idx = i;
      }
    }

    std::cout << "Smallest strictly positive eigenvalue: " << second_smallest
              << std::endl;

    if (second_smallest_idx >= 0) {
      Eigen::VectorXd fiedler = eigenvectors.col(second_smallest_idx);
      std::cout << "Fielder vector: " << std::endl;
      for (int i = 0; i < fiedler.size(); i++) {
        std::cout << " Node " << (i + 1) << ": " << fiedler(i)
                  << (fiedler(i) > 0 ? "(positive)" : "(negative)")
                  << std::endl;
      }
    }
  }

  void load_social_network_matrix(const std::string &filename) {
    bool _loaded = Eigen::loadMarket(As, filename);

    std::cout << "Matrix properties:" << std::endl;
    std::cout << "  Dimensions: " << As.rows() << "x" << As.cols() << std::endl;

    std::cout << "  Frobenius norm (Eigen's norm()): " << As.norm()
              << std::endl;
  }

  void create_social_network_laplasian() {
    int n = As.rows();
    vs = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < n; i++) {
      vs(i) = As.row(i).sum();
    }

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; i++) {
      triplets.push_back(Eigen::Triplet<double>(i, i, vs(i)));
    }

    Ds.resize(n, n);
    Ds.setFromTriplets(triplets.begin(), triplets.end());

    Ls = Ds - As;

    Eigen::SparseMatrix<double> LsT = Ls.transpose();
    Eigen::SparseMatrix<double> diff = Ls - LsT;
    bool is_symmetric = diff.norm() < 1e-10;

    std::cout << "Is Ls symmetric? " << (is_symmetric ? "Yes" : "No")
              << std::endl;
    std::cout << "Number of nonzero entries in Ls: " << Ls.nonZeros()
              << std::endl;
  }

  void find_largest_eigenvalues() {
    Ls.coeffRef(0, 0) += 0.2;

    Eigen::saveMarket(Ls, "Ls_perturbed.mtx");
    std::cout << "Exported perturbed Laplacian to Ls_perturbed.mtx"
              << std::endl;

    LIS_MATRIX A;
    LIS_VECTOR x;
    LIS_ESOLVER esolver;
    LIS_INT n = Ls.rows();
    LIS_INT iter;
    LIS_REAL evalue;

    lis_matrix_create(LIS_COMM_WORLD, &A);
    lis_matrix_set_size(A, n, 0);

    for (int k = 0; k < Ls.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Ls, k); it; ++it) {
        lis_matrix_set_value(LIS_INS_VALUE, it.row(), it.col(), it.value(), A);
      }
    }

    lis_matrix_assemble(A);

    lis_vector_create(LIS_COMM_WORLD, &x);
    lis_vector_set_size(x, n, 0);

    lis_vector_set_all(1.0, x);

    lis_esolver_create(&esolver);

    lis_esolver_set_option((char *)"-e pi -etol 1e-8 -emaxiter 10000", esolver);

    lis_esolve(A, x, &evalue, esolver);

    lis_esolver_get_iter(esolver, &iter);

    std::cout << "Largest eigenvalue: " << evalue << std::endl;
    std::cout << "Iterations: " << iter << std::endl;

    lis_esolver_destroy(esolver);
    lis_matrix_destroy(A);
    lis_vector_destroy(x);
  }
};

int main(int argc, char *argv[]) {
  lis_initialize(&argc, &argv);

  ChallengeTwo c2;

  // Point 1
  c2.create_small_graph();

  // Point 2
  c2.create_graph_laplasian();

  // Point 3
  c2.find_eigenvalues();

  // Point 4
  c2.find_fielder_vector();

  // Point 5
  c2.load_social_network_matrix("assets/social.mtx");

  // Point 6
  c2.create_social_network_laplasian();

  // Point 7
  c2.find_largest_eigenvalues();

  lis_finalize();

  return 0;
}
