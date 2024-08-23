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
  std::vector<std::pair<float, float>> points, queries;
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

  MPI_Bcast(points.data(), N * 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(queries.data(), M * 2, MPI_INT, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, M); // in case WORLD_SIZE > M
  int numQueriesPerProc = (int)std::ceil((float)M / NUM_PROC);
  int start = numQueriesPerProc * WORLD_RANK,
      end = std::min(M, numQueriesPerProc * (WORLD_RANK + 1));

  const int pointsPerQuery = std::min(K, N);

  std::vector<std::vector<std::pair<float, float>>> localNN;

  if (WORLD_RANK >= NUM_PROC)
    goto end;

  localNN.resize(end - start);
  for (int i = start; i < end; i++) {
    const std::pair<float, float> &query = queries[i];
    auto cmp = [&query](const std::pair<float, float> &a,
                        const std::pair<float, float> &b) -> bool {
      return std::pow(a.first - query.first, 2) +
                 std::pow(a.second - query.second, 2) <
             std::pow(b.first - query.first, 2) +
                 std::pow(b.second - query.second, 2);
    };
    std::priority_queue<std::pair<float, float>,
                        std::vector<std::pair<float, float>>, decltype(cmp)>
        pq(cmp);
    for (const std::pair<float, float> &point : points) {
      pq.push(point);
      if (pq.size() > pointsPerQuery)
        pq.pop();
    }

    std::vector<std::pair<float, float>> &nn = localNN[i - start];
    while (!pq.empty()) {
      nn.push_back(pq.top());
      pq.pop();
    }
  }

  if (WORLD_RANK != 0) {
    for (int i = 0; i < localNN.size(); i++)
      MPI_Send(localNN[i].data(), pointsPerQuery * 2, MPI_INT, 0, 0,
               MPI_COMM_WORLD);
  } else {
    std::vector<std::vector<std::pair<float, float>>> globalNN(
        M, std::vector<std::pair<float, float>>(pointsPerQuery));

    // push the current local nearest neighbours
    for (int i = 0; i < localNN.size(); i++)
      globalNN[i] = localNN[i];

    // receive the nearest neighbours from other processes
    for (int i = 1; i < NUM_PROC; i++) {
      int start = numQueriesPerProc * i,
          end = std::min(M, numQueriesPerProc * (i + 1));
      for (int j = start; j < end; j++) {
        MPI_Recv(globalNN[j].data(), pointsPerQuery * 2, MPI_INT, i, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
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
