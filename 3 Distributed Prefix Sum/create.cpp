#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

std::vector<double> generate_test_case(int n) {
  std::vector<double> numbers(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 10.0);

  for (int i = 0; i < n; i++) {
    numbers[i] = dis(gen);
  }
  return numbers;
}

std::vector<double> calculate_prefix_sums(const std::vector<double> &numbers) {
  std::vector<double> prefix_sums(numbers.size());
  double sum = 0;
  for (size_t i = 0; i < numbers.size(); i++) {
    sum += numbers[i];
    prefix_sums[i] = sum;
  }
  return prefix_sums;
}

void write_files(int n) {
  auto numbers = generate_test_case(n);

  // Write input file
  std::ofstream numbers_file("random.txt");

  numbers_file << std::setprecision(10);

  numbers_file << n << "\n";
  for (const auto &num : numbers) {
    numbers_file << num << " ";
  }
  numbers_file << "\n";
  numbers_file.close();

  // read from input file
  numbers.clear();
  std::ifstream data_file("random.txt");

  data_file >> n;
  for (int i = 0; i < n; i++) {
    double num;
    data_file >> num;
    numbers.push_back(num);
  }

  // Write output file
  auto prefix_sums = calculate_prefix_sums(numbers);
  std::ofstream output_file("random-opt.txt");

  output_file << std::setprecision(10);

  for (const auto &sum : prefix_sums) {
    output_file << sum << " ";
  }
  output_file << std::endl;
  output_file.close();
}

int main() {
  int n = 100000; // Set n manually here
  write_files(n);
  return 0;
}