#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main() {
  // Matrix operations
  Eigen::Matrix3d A;
  A << 1, 2, 3, 4, 5, 6, 7, 8, 10;

  Eigen::Vector3d b(3, 3, 4);

  cout << "Matrix A:\n" << A << endl;
  cout << "Vector b:\n" << b << endl;

  // Solve Ax = b
  Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
  cout << "Solution x:\n" << x << endl;

  // Matrix multiplication
  Eigen::Matrix3d C = A * A.transpose();
  cout << "A * A^T:\n" << C << endl;

  // Eigenvalues and eigenvectors
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(C);
  cout << "Eigenvalues:\n" << eigensolver.eigenvalues() << endl;

  return 0;
}
