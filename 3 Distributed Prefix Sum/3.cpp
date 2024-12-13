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
                                            std::chrono::steady_clock::now(),
                                        endTime;

  // broadcast the number of elements to all processes
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, N);

  auto getBounds = [&NUM_PROC, &N](int rank) -> std::pair<int, int> {
    const int baseItems = N / NUM_PROC;
    const int extraItems = N % NUM_PROC;
    const int start = baseItems * rank + std::min(rank, extraItems),
              end = start + baseItems + (rank < extraItems ? 1 : 0);

    return {start, end};
  };

  std::vector<double> localNums;

  const auto [start, end] = getBounds(WORLD_RANK);

  if (WORLD_RANK >= NUM_PROC)
    goto finalise;

  // store local numbers for each process to handle
  if (WORLD_RANK == 0) {
    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i);
      std::vector<double> numsToSend(numbers.begin() + qStart,
                                     numbers.begin() + qEnd);
      MPI_Send(numsToSend.data(), numsToSend.size(), MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
    }

    localNums =
        std::vector<double>(numbers.begin() + start, numbers.begin() + end);
    numbers.clear();
  } else {
    localNums.resize(end - start);
    MPI_Recv(localNums.data(), end - start, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  // calculate the prefix sums within the localNums array
  for (int i = 1; i < localNums.size(); i++)
    localNums[i] += localNums[i - 1];

  // send closing values to rank 0
  // in rank 0, find values to add for remaining processes
  if (WORLD_RANK != 0) {
    MPI_Send(&localNums.back(), 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

    double prefixWeight = 0;
    MPI_Recv(&prefixWeight, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (int i = 0; i < localNums.size(); i++)
      localNums[i] += prefixWeight;
  } else {
    std::vector<double> closingNums(NUM_PROC);
    closingNums[0] = localNums.back();

    for (int i = 1; i < NUM_PROC; i++) {
      MPI_Recv(&closingNums[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    for (int i = 1; i < closingNums.size(); i++)
      closingNums[i] += closingNums[i - 1];

    for (int i = 1; i < NUM_PROC; i++)
      MPI_Send(&closingNums[i - 1], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
  }

  // send the localNums back to rank 0 and accumulate
  if (WORLD_RANK != 0) {
    MPI_Send(localNums.data(), localNums.size(), MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD);
  } else {
    std::vector<double> globalNums(N);
    for (int i = 0; i < localNums.size(); i++)
      globalNums[i] = localNums[i];

    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i);
      MPI_Recv(globalNums.data() + qStart, qEnd - qStart, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    endTime = std::chrono::steady_clock::now();

    std::string fileName = std::to_string(N) + "_time.txt";

    std::ofstream output(fileName, std::ios::app);
    output << WORLD_SIZE << ":"
           << std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                   beginTime)
                  .count()
           << std::endl;

    for (int i = 0; i < globalNums.size(); i++)
      std::cout << globalNums[i] << " ";
    std::cout << std::endl;
  }

finalise:
  MPI_Finalize();
  return 0;
}
