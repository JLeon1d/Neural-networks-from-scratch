#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(3,3);
    std::cout << m << std::endl;
    std::cout << "WORKS\n";
    return 0;
}