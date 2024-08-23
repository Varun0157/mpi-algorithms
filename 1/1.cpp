/*
Distributed K Nearest Neighbours
*/

#include <algorithm>
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

  std::vector<std::vector<std::pair<float, float>>> localNN;

  const int NUM_PROC = std::min(WORLD_SIZE, M); // in case WORLD_SIZE > M
  int numQueriesPerProc = (int)std::ceil((float)M / NUM_PROC);
  int start = numQueriesPerProc * WORLD_RANK,
      end = std::min(M, numQueriesPerProc * (WORLD_RANK + 1));

  const int pointsPerQuery = std::min(K, N);

  for (int i = start; i < end; i++) {
    std::vector<std::pair<int, std::pair<float, float>>> dist;
    for (int j = 0; j < N; j++) {
      int d = (queries[i].first - points[j].first) *
                  (queries[i].first - points[j].first) +
              (queries[i].second - points[j].second) *
                  (queries[i].second - points[j].second);
      dist.push_back({d, points[j]});
    }
    std::sort(dist.begin(), dist.end());
    std::vector<std::pair<float, float>> nn;
    for (int j = 0; j < pointsPerQuery; j++)
      nn.push_back(dist[j].second);
    localNN.push_back(nn);
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

  MPI_Finalize();
  return 0;
}
