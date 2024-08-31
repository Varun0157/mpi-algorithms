/*
Distributed prefix sum
*/

#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <input file>" << std::endl;
    return 1;
  }

  MPI_Init(&argc, &argv);

  int WORLD_SIZE;
  MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);

  int WORLD_RANK;
  MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);

  int N;
  std::vector<double> numbers;
  if (WORLD_RANK == 0) {
    std::ifstream input(argv[1]);

    input >> N;

    numbers.resize(N);
    for (double &number : numbers)
      input >> number;
  }

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

  if (WORLD_RANK != 0) {
    double last = localNums.back();
    MPI_Send(&last, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

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

    for (int i = 0; i < globalNums.size(); i++)
      std::cout << globalNums[i] << " ";
    std::cout << std::endl;
  }

finalise:
  MPI_Finalize();
  return 0;
}
