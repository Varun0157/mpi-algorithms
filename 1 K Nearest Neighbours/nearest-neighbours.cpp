/*
Distributed K Nearest Neighbours
*/

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <queue>
#include <vector>

int finalise() {
  MPI_Finalize();
  return 0;
}

void getPointsAndQueries(const char *filename, int &N, int &M, int &K,
                         std::vector<std::pair<double, double>> &points,
                         std::vector<std::pair<double, double>> &queries) {
  std::ifstream input(filename);
  if (!input) {
    std::cerr << "error opening file" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  input >> N >> M >> K;

  points.resize(N);
  for (int i = 0; i < N; i++) {
    double x, y;
    input >> x >> y;
    points[i] = {x, y};
  }
  queries.resize(M);
  for (int i = 0; i < M; i++)
    input >> queries[i].first >> queries[i].second;

  input.close();
}

std::pair<int, int> getBounds(int rank, int NUM_PROC, int M) {
  const int baseItems = M / NUM_PROC;
  const int extraItems = M % NUM_PROC;
  const int start = baseItems * rank + std::min(rank, extraItems),
            end = start + baseItems + (rank < extraItems ? 1 : 0);

  return {start, end};
}

std::vector<std::vector<std::pair<double, double>>>
getLocalNearestNeighbours(std::vector<std::pair<double, double>> &points,
                          std::vector<std::pair<double, double>> &localQueries,
                          const int WORLD_RANK, const int NUM_PROC, const int M,
                          int pointsPerQuery) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, M);

  std::vector<std::vector<std::pair<double, double>>> localNN(end - start);
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

  return localNN;
}

void sendLocalQueries(std::vector<std::pair<double, double>> &queries,
                      const int NUM_PROC, const int M, const int PARENT_RANK) {
  for (int i = 0; i < NUM_PROC; i++) {
    if (i == PARENT_RANK)
      continue;

    const auto [qStart, qEnd] = getBounds(i, NUM_PROC, M);
    std::vector<std::pair<double, double>> procQueries(queries.begin() + qStart,
                                                       queries.begin() + qEnd);
    MPI_Send(procQueries.data(), procQueries.size() * 2, MPI_DOUBLE, i, 0,
             MPI_COMM_WORLD);
  }
}

void setLocalQueries(const int WORLD_RANK, const int WORLD_SIZE, const int M,
                     const int NUM_PROC,
                     std::vector<std::pair<double, double>> &queries,
                     std::vector<std::pair<double, double>> &localQueries) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, M);

  // gather local queries for each process to handle
  if (WORLD_RANK == 0) {
    sendLocalQueries(queries, NUM_PROC, M, 0);

    localQueries = std::vector<std::pair<double, double>>(
        queries.begin() + start, queries.begin() + end);
    queries.clear();
  } else {
    localQueries.resize(end - start);
    MPI_Recv(localQueries.data(), (end - start) * 2, MPI_DOUBLE, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

std::chrono::time_point<std::chrono::steady_clock> accumulateNearestNeighbours(
    const int WORLD_RANK, const int WORLD_SIZE, const int NUM_PROC, const int M,
    const int pointsPerQuery,
    std::vector<std::vector<std::pair<double, double>>> &localNN,
    std::vector<std::vector<std::pair<double, double>>> &globalNN) {
  // send and accumulate local nearest neighbours at rank 0
  if (WORLD_RANK != 0) {
    for (int i = 0; i < localNN.size(); i++)
      MPI_Send(localNN[i].data(), pointsPerQuery * 2, MPI_DOUBLE, 0, 0,
               MPI_COMM_WORLD);
  } else {
    globalNN.resize(M, std::vector<std::pair<double, double>>(pointsPerQuery));

    // push the current local nearest neighbours (rank 0)
    for (int i = 0; i < localNN.size(); i++)
      globalNN[i] = localNN[i];

    // receive the nearest neighbours from other processes
    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i, NUM_PROC, M);
      for (int j = qStart; j < qEnd; j++)
        MPI_Recv(globalNN[j].data(), pointsPerQuery * 2, MPI_DOUBLE, i, 0,
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

  int N, M, K;
  std::vector<std::pair<double, double>> points, queries;
  if (WORLD_RANK == 0) {
    getPointsAndQueries(argv[1], N, M, K, points, queries);
  }

  std::chrono::steady_clock::time_point beginTime =
      std::chrono::steady_clock::now();

  // broadcast N, M, K and all points to all processes
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (WORLD_RANK != 0)
    points.resize(N);
  MPI_Bcast(points.data(), N * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, M); // in case WORLD_SIZE > M

  if (WORLD_RANK >= NUM_PROC)
    return finalise();

  const int pointsPerQuery = std::min(K, N);

  std::vector<std::pair<double, double>> localQueries;
  setLocalQueries(WORLD_RANK, WORLD_SIZE, M, NUM_PROC, queries, localQueries);
  auto localNN = getLocalNearestNeighbours(points, localQueries, WORLD_RANK,
                                           NUM_PROC, M, pointsPerQuery);

  std::vector<std::vector<std::pair<double, double>>> globalNN;
  auto endTime = accumulateNearestNeighbours(
      WORLD_RANK, WORLD_SIZE, NUM_PROC, M, pointsPerQuery, localNN, globalNN);

  if (WORLD_RANK != 0)
    return finalise();

  endTime = std::chrono::steady_clock::now();

  // create a file named {N}_{M}_{K}_time.txt
  std::string fileName = std::to_string(N) + "_" + std::to_string(M) + "_" +
                         std::to_string(K) + "_time.txt";
  std::ofstream output(fileName, std::ios::app);
  output << WORLD_SIZE << ":"
         << std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                 beginTime)
                .count()
         << std::endl;

  // print out the nearest neighbours
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < globalNN[i].size(); j++)
      std::cout << globalNN[i][j].first << " " << globalNN[i][j].second
                << std::endl;
  }

  return finalise();
}
