#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

class ChallengeOne {
private:
  Eigen::MatrixXd original_image;
  Eigen::MatrixXd noisy_image;
  int rows, cols;
  Eigen::SparseMatrix<double> a2, a3;

public:
  bool load_image(const std::string &filename) {
    int width, height, channels;

    unsigned char *data =
        stbi_load(filename.c_str(), &width, &height, &channels, 1);

    if (!data) {
      std::cerr << "failed to load image" << filename << std::endl;
      return false;
    }

    rows = height;
    cols = width;
    original_image = Eigen::MatrixXd(rows, cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        original_image(i, j) = static_cast<double>(data[i * width + j]);
      }
    }

    stbi_image_free(data);
    std::cout << "image loaded: " << rows << " x " << cols << " pixels"
              << std::endl;

    return true;
  }

  void save_image(const Eigen::MatrixXd &matrix, const std::string &filename) {
    int height = matrix.rows();
    int width = matrix.cols();
    std::vector<unsigned char> image_data(height * width);

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        double val = matrix(i, j);
        val = std::max(0.0, std::min(255.0, val));
        image_data[i * width + j] = static_cast<unsigned char>(val);
      }
    }

    stbi_write_png(filename.c_str(), width, height, 1, image_data.data(),
                   width);

    std::cout << "image saved: " << filename << std::endl;
  }

  void add_noise() {
    Eigen::MatrixXd noise = Eigen::MatrixXd::Random(rows, cols) * 40;
    noisy_image = original_image + noise;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        noisy_image(i, j) = std::max(0.0, std::min(255.0, noisy_image(i, j)));
      }
    }

    save_image(noisy_image, "noisy_image.png");
  }

  Eigen::VectorXd reshape_image_to_vector(const Eigen::MatrixXd &image) {
    Eigen::VectorXd vec(rows * cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        vec(i * cols + j) = image(i, j);
      }
    }
    return vec;
  }

  void print_stats() {
    Eigen::VectorXd v = reshape_image_to_vector(original_image);
    Eigen::VectorXd w = reshape_image_to_vector(noisy_image);

    std::cout << "Image dimensions: " << rows << " x " << cols << std::endl;
    std::cout << "Vector size: " << v.size() << " components" << std::endl;

    std::cout << "Euclidean norm of original image vector: " << v.norm()
              << std::endl;
    std::cout << "Euclidean norm of noisy image vector: " << w.norm()
              << std::endl;
  }

  Eigen::SparseMatrix<double>
  build_convolution_matrix(const Eigen::MatrixXd &kernel) {
    const int h = rows, w = cols;                     // Image dimensions
    const int kh = kernel.rows(), kw = kernel.cols(); // Kernel dimensions
    const int cy = kh / 2, cx = kw / 2;               // Kernel center
    const int n = h * w;                              // Total pixels

    auto flatten = [w](int y, int x) { return y * w + x; };

    auto in_bounds = [h, w](int y, int x) {
      return y >= 0 && y < h && x >= 0 && x < w;
    };

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * kh * kw);

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        int out_idx = flatten(y, x);

        for (int ky = 0; ky < kh; ky++) {
          for (int kx = 0; kx < kw; kx++) {
            int in_y = y + ky - cy;
            int in_x = x + kx - cx;

            if (in_bounds(in_y, in_x)) {
              double weight = kernel(ky, kx);
              if (weight != 0) {
                triplets.emplace_back(out_idx, flatten(in_y, in_x), weight);
              }
            }
          }
        }
      }
    }

    Eigen::SparseMatrix<double> a(n, n);
    a.setFromTriplets(triplets.begin(), triplets.end());
    return a;
  }

  Eigen::MatrixXd vector_to_image(const Eigen::VectorXd &vec) {
    Eigen::MatrixXd img(rows, cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        img(i, j) = vec(i * cols + j);
      }
    }

    return img;
  }

  void apply_smoothing_filter() {
    Eigen::Matrix3d hav1;
    hav1 << 1, 1, 0, 1, 2, 1, 0, 1, 1;
    hav1 /= 8.0;

    Eigen::SparseMatrix<double> a1 = build_convolution_matrix(hav1);

    std::cout << "Smoothing matrix A1 non-zero entries: " << a1.nonZeros()
              << std::endl;

    Eigen::VectorXd w = reshape_image_to_vector(noisy_image);
    Eigen::VectorXd smoothed = a1 * w;

    Eigen::MatrixXd smoothed_img = vector_to_image(smoothed);
    save_image(smoothed_img, "smoothed_image.png");
  }

  void apply_sharpening_filter() {
    Eigen::Matrix3d hsh1;
    hsh1 << 0, -2, 0, -2, 9, -2, 0, -2, 0;

    Eigen::SparseMatrix<double> a2_temp = build_convolution_matrix(hsh1);

    // Point 6
    std::cout << "Sharpening matrix A2 non-zero entries: " << a2_temp.nonZeros()
              << std::endl;

    // Point 7
    Eigen::SparseMatrix<double> a2_transpose = a2_temp.transpose();
    bool symmetric = (a2_temp - a2_transpose).norm() < 1e-10;
    std::cout << "Is A2 symmetric? " << (symmetric ? "Yes" : "No") << std::endl;

    // Point 8
    Eigen::VectorXd v = reshape_image_to_vector(original_image);
    Eigen::VectorXd sharpened = a2_temp * v;

    Eigen::MatrixXd sharpened_img = vector_to_image(sharpened);
    save_image(sharpened_img, "sharpened_image.png");

    this->a2 = a2_temp;
  }

  void export_sparse_matrix_to_mtx_format(const Eigen::SparseMatrix<double> &mat,
                                          const std::string &filename) {
    std::ofstream file(filename);
    file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    file << mat.rows() << " " << mat.cols() << " " << mat.nonZeros()
         << std::endl;

    for (int k = 0; k < mat.outerSize(); k++) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
        file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value()
             << std::endl;
      }
    }
    file.close();
    std::cout << "Matrix exported to " << filename << std::endl;
  }

  void export_vector_to_mtx(const Eigen::VectorXd &vec,
                            const std::string &filename) {
    std::ofstream file(filename);

    file << "MatrixMarket matrix coordinate real general" << std::endl;
    file << vec.size() << " 1 " << vec.size() << std::endl;

    for (int i = 0; i < vec.size(); ++i) {
      if (std::abs(vec(i)) > 1e-10) {
        file << (i + 1) << " 1 " << vec(i) << std::endl;
      }
    }

    file.close();
    std::cout << "Vector exported to " << filename << std::endl;
  }

  void apply_edge_detection_filter() {
    Eigen::Matrix3d hed2;
    hed2 << -1, -2, -1, 0, 0, 0, 1, 2, 1;

    Eigen::SparseMatrix<double> a3_temp = build_convolution_matrix(hed2);

    Eigen::SparseMatrix<double> a3_transpose = a3_temp.transpose();
    bool symmetric = (a3_temp - a3_transpose).norm() < 1e-10;
    std::cout << "Is A3 symmetric? " << (symmetric ? "Yes" : "No") << std::endl;

    Eigen::VectorXd v = reshape_image_to_vector(original_image);
    Eigen::VectorXd edges = a3_temp * v;

    Eigen::MatrixXd edges_img = vector_to_image(edges);

    save_image(edges_img, "edge_detection.png");

    this->a3 = a3_temp;
  }

  void solve_with_eigen() {
    Eigen::VectorXd w = reshape_image_to_vector(noisy_image);
    int n = rows * cols;

    // Point 13
    Eigen::SparseMatrix<double> i(n, n);
    i.setIdentity();
    Eigen::SparseMatrix<double> system_matrix = 3.0 * i + a3;

    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.setTolerance(1e-8);
    solver.compute(system_matrix);

    Eigen::VectorXd y = solver.solve(w);

    std::cout << "Solve for (3I + A3)*y = w:" << std::endl;
    std::cout << "Iterations: " << solver.iterations() << std::endl;
    std::cout << "Final residual: " << (system_matrix * y - w).norm()
              << std::endl;

    // Point 14
    Eigen::MatrixXd result_img = vector_to_image(y);
    save_image(result_img, "eigen_solution.png");
  }
};

int main() {
  ChallengeOne c;

  // Point 1
  if (!c.load_image("assets/uma.jpg")) {
    return 1;
  }

  // Point 2
  c.add_noise();

  // Point 3
  c.print_stats();

  // Point 4 + 5
  c.apply_smoothing_filter();

  // Point 6 + 7 + 8
  c.apply_sharpening_filter();

  // Point 9, 10
  // TODO: Use LIS lib for solving the linear system

  // Point 11, 12
  c.apply_edge_detection_filter();

  // Point 13, 14
  c.solve_with_eigen();

  return 0;
}
