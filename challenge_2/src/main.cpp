#include <Eigen/Eigen>
#include <iostream>
#include <lis.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

class ChallengeTwo {
private:
  Eigen::MatrixXd Ag;
  Eigen::VectorXd vg;
  Eigen::MatrixXd Dg;
  Eigen::MatrixXd Lg;

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
    std::cout << "Comment: One eigenvalue is zero (graph is connected), others "
                 "are positive."
              << std::endl;
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
};

int main() {

  ChallengeTwo c2;

  // Point 1
  c2.create_small_graph();

  // Point 2
  c2.create_graph_laplasian();

  // Point 3
  c2.find_eigenvalues();

	// Point 4
	c2.find_fielder_vector();

  return 0;
}
