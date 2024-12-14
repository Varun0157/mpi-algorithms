/*
Distributed prefix sum
*/

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

int finalise() {
  MPI_Finalize();
  return 0;
}

void getNumbers(const char *filename, int &N, std::vector<double> &numbers) {
  std::ifstream input(filename);
  if (!input) {
    std::cerr << "error opening file" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  input >> N;

  numbers.resize(N);
  for (double &number : numbers)
    input >> number;

  input.close();
}

std::pair<int, int> getBounds(const int rank, const int NUM_PROC, const int N) {
  const int baseItems = N / NUM_PROC;
  const int extraItems = N % NUM_PROC;
  const int start = baseItems * rank + std::min(rank, extraItems),
            end = start + baseItems + (rank < extraItems ? 1 : 0);

  return {start, end};
}

void sendLocalNumbersFromZero(const int NUM_PROC, const int N,
                              const std::vector<double> &numbers) {
  for (int i = 1; i < NUM_PROC; i++) {
    const auto [qStart, qEnd] = getBounds(i, NUM_PROC, N);
    std::vector<double> numsToSend(numbers.begin() + qStart,
                                   numbers.begin() + qEnd);
    MPI_Send(numsToSend.data(), numsToSend.size(), MPI_DOUBLE, i, 0,
             MPI_COMM_WORLD);
  }
}

void storeLocalPrefixSums(std::vector<double> &nums) {
  for (int i = 1; i < nums.size(); i++)
    nums[i] += nums[i - 1];
}

void setLocalNums(const int WORLD_RANK, const int WORLD_SIZE, const int N,
                  const int NUM_PROC, std::vector<double> &numbers,
                  std::vector<double> &localNums) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, N);

  // store local numbers for each process to handle
  if (WORLD_RANK == 0) {
    sendLocalNumbersFromZero(NUM_PROC, N, numbers);

    localNums =
        std::vector<double>(numbers.begin() + start, numbers.begin() + end);
    numbers.clear();
  } else {
    localNums.resize(end - start);
    MPI_Recv(localNums.data(), end - start, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void addWeightFromPreviousSets(const int WORLD_RANK, const int NUM_PROC,
                               std::vector<double> &localNums) {
  if (WORLD_RANK != 0) {
    // send closing value to rank 0
    MPI_Send(&localNums.back(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    // receive value to add from rank 0
    double prefixWeight = 0;
    MPI_Recv(&prefixWeight, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (int i = 0; i < localNums.size(); i++)
      localNums[i] += prefixWeight;
  } else {
    std::vector<double> closingNums(NUM_PROC);
    closingNums[0] = localNums.back();

    // receive closing nums from each remaining process
    for (int i = 1; i < NUM_PROC; i++) {
      MPI_Recv(&closingNums[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    // calculate prefix weight for each process
    for (int i = 1; i < closingNums.size(); i++)
      closingNums[i] += closingNums[i - 1];

    // send prefix weights to each process
    for (int i = 1; i < NUM_PROC; i++)
      MPI_Send(&closingNums[i - 1], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
  }
}

std::chrono::time_point<std::chrono::steady_clock>
accumulateLocalSums(const int WORLD_RANK, const int NUM_PROC, const int N,
                    std::vector<double> &localNums,
                    std::vector<double> &globalNums) {
  // send the localNums back to rank 0 and accumulate
  if (WORLD_RANK != 0) {
    MPI_Send(localNums.data(), localNums.size(), MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD);
  } else {
    globalNums.resize(N);
    for (int i = 0; i < localNums.size(); i++)
      globalNums[i] = localNums[i];

    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i, NUM_PROC, N);
      MPI_Recv(globalNums.data() + qStart, qEnd - qStart, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  return std::chrono::steady_clock::now();
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <input file>" << std::endl;
    return 1;
  }

  std::cout << std::setprecision(10);

  MPI_Init(&argc, &argv);

  int WORLD_SIZE;
  MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);

  int WORLD_RANK;
  MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);

  int N;
  std::vector<double> numbers;
  if (WORLD_RANK == 0) {
    getNumbers(argv[1], N, numbers);
  }

  std::chrono::steady_clock::time_point beginTime =
      std::chrono::steady_clock::now();

  // broadcast the number of elements to all processes
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, N);

  std::vector<double> localNums;

  if (WORLD_RANK >= NUM_PROC)
    return finalise();

  setLocalNums(WORLD_RANK, WORLD_SIZE, N, NUM_PROC, numbers, localNums);
  storeLocalPrefixSums(localNums);
  addWeightFromPreviousSets(WORLD_RANK, NUM_PROC, localNums);

  std::vector<double> globalNums;
  auto endTime =
      accumulateLocalSums(WORLD_RANK, NUM_PROC, N, localNums, globalNums);

  if (WORLD_RANK != 0)
    return finalise();

  std::ofstream output(std::to_string(N) + "_time.txt", std::ios::app);
  output << WORLD_SIZE << ":"
         << std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                 beginTime)
                .count()
         << std::endl;

  for (const double &num : globalNums)
    std::cout << num << " ";
  std::cout << std::endl;

  return finalise();
}
