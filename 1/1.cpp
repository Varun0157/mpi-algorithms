/*
Distributed K Nearest Neighbours
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <queue>
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

  int N, M, K;
  std::vector<std::pair<double, double>> points, queries;
  if (WORLD_RANK == 0) {
    std::ifstream input(argv[1]);

    input >> N >> M >> K;

    points.resize(N);
    for (int i = 0; i < N; i++)
      input >> points[i].first >> points[i].second;
    queries.resize(M);
    for (int i = 0; i < M; i++)
      input >> queries[i].first >> queries[i].second;
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (WORLD_RANK != 0)
    points.resize(N);
  MPI_Bcast(points.data(), N * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, M); // in case WORLD_SIZE > M
  const int QUERIES_PER_PROC = (int)std::ceil((double)M / NUM_PROC);

  auto getBounds = [&NUM_PROC, &QUERIES_PER_PROC,
                    &M](int rank) -> std::pair<int, int> {
    int start = QUERIES_PER_PROC * rank,
        end = std::min(M, QUERIES_PER_PROC * (rank + 1));
    return {start, end};
  };

  std::vector<std::pair<double, double>> localQueries;
  std::vector<std::vector<std::pair<double, double>>> localNN;

  const auto [start, end] = getBounds(WORLD_RANK);
  const int pointsPerQuery = std::min(K, N);

  if (WORLD_RANK >= NUM_PROC)
    goto finalise;

  if (WORLD_RANK == 0) {
    // broadcast the queries to the required processes
    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i);
      std::vector<std::pair<double, double>> procQueries(
          queries.begin() + qStart, queries.begin() + qEnd);
      MPI_Send(procQueries.data(), procQueries.size() * 2, MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
    }

    localQueries = std::vector<std::pair<double, double>>(
        queries.begin() + start, queries.begin() + end);
    queries.clear();
  } else {
    localQueries.resize(end - start);
    MPI_Recv(localQueries.data(), (end - start) * 2, MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // print the local queries
  // std::cout << WORLD_RANK << std::endl;
  // for (int i = 0; i < localQueries.size(); i++)
  //   std::cout << localQueries[i].first << " " << localQueries[i].second
  //             << std::endl;

  localNN.resize(end - start);
  for (int i = 0; i < localQueries.size(); i++) {
    const std::pair<double, double> &query = localQueries[i];
    auto cmp = [&query](const std::pair<double, double> &a,
                        const std::pair<double, double> &b) -> bool {
      const double distA = std::pow(a.first - query.first, 2) +
                           std::pow(a.second - query.second, 2),
                   distB = std::pow(b.first - query.first, 2) +
                           std::pow(b.second - query.second, 2);

      return distA < distB or (distA == distB and a > b);
    };
    std::priority_queue<std::pair<double, double>,
                        std::vector<std::pair<double, double>>, decltype(cmp)>
        pq(cmp);
    for (const std::pair<double, double> &point : points) {
      pq.push(point);
      if (pq.size() > pointsPerQuery)
        pq.pop();
    }

    std::vector<std::pair<double, double>> &nn = localNN[i];
    while (!pq.empty()) {
      nn.push_back(pq.top());
      pq.pop();
    }
  }

  if (WORLD_RANK != 0) {
    for (int i = 0; i < localNN.size(); i++)
      MPI_Send(localNN[i].data(), pointsPerQuery * 2, MPI_DOUBLE, 0, 0,
               MPI_COMM_WORLD);
  } else {
    std::vector<std::vector<std::pair<double, double>>> globalNN(
        M, std::vector<std::pair<double, double>>(pointsPerQuery));

    // push the current local nearest neighbours
    for (int i = 0; i < localNN.size(); i++)
      globalNN[i] = localNN[i];

    // receive the nearest neighbours from other processes
    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i);
      for (int j = qStart; j < qEnd; j++)
        MPI_Recv(globalNN[j].data(), pointsPerQuery * 2, MPI_DOUBLE, i, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // print out the nearest neighbours
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < globalNN[i].size(); j++)
        std::cout << globalNN[i][j].first << " " << globalNN[i][j].second
                  << std::endl;
    }
  }

finalise:
  MPI_Finalize();
  return 0;
}
