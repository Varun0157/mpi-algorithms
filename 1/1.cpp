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

  if (WORLD_RANK != 0) {
    points.resize(N);
    queries.resize(M);
  }

  MPI_Bcast(points.data(), N * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(queries.data(), M * 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, M); // in case WORLD_SIZE > M
  int numQueriesPerProc = (int)std::ceil((double)M / NUM_PROC);

  auto getBounds = [&NUM_PROC, &numQueriesPerProc,
                    &M](int rank) -> std::pair<int, int> {
    int start = numQueriesPerProc * rank,
        end = std::min(M, numQueriesPerProc * (rank + 1));
    return {start, end};
  };

  const auto [start, end] = getBounds(WORLD_RANK);

  const int pointsPerQuery = std::min(K, N);

  std::vector<std::vector<std::pair<double, double>>> localNN;

  if (WORLD_RANK >= NUM_PROC)
    goto end;

  localNN.resize(end - start);
  for (int i = start; i < end; i++) {
    const std::pair<double, double> &query = queries[i];
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

    std::vector<std::pair<double, double>> &nn = localNN[i - start];
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
      const auto [start, end] = getBounds(i);
      for (int j = start; j < end; j++)
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

end:
  MPI_Finalize();
  return 0;
}
