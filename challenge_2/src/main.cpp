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
};

int main() {

  ChallengeTwo c2;

  // Point 1
  c2.create_small_graph();

  return 0;
}
