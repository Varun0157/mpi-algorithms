/*
Parallel matrix inverse using row reduction method and MPI
*/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

void printVector(const std::vector<double> &vec) {
  for (const auto &el : vec) {
    std::cout << el << " ";
  }
  std::cout << std::endl;
}

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
  std::vector<std::vector<double>> matrix, identity;
  if (WORLD_RANK == 0) {
    std::ifstream input(argv[1]);

    input >> N;

    matrix.resize(N);
    identity.resize(N);
    for (int i = 0; i < N; i++) {
      matrix[i].resize(N);
      identity[i].resize(N);
      for (int j = 0; j < N; j++) {
        input >> matrix[i][j];
        identity[i][j] = (double)i == j;
      }
    }
  }

  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, N);

  auto getBounds = [&NUM_PROC, &N](int rank) -> std::pair<int, int> {
    const int baseItems = N / NUM_PROC;
    const int extraItems = N % NUM_PROC;

    const int start = rank * baseItems + std::min(rank, extraItems);
    const int end = start + baseItems + (rank < extraItems ? 1 : 0);

    return {start, end};
  };

  auto findRank = [&NUM_PROC, &N](int index) -> int {
    const int baseItems = N / NUM_PROC;
    const int extraItems = N % NUM_PROC;

    int rank = (index < (baseItems + 1) * extraItems)
                   ? index / (baseItems + 1)
                   : (index - extraItems) / baseItems;

    return rank;
  };

  std::vector<std::vector<double>> localMatrix, localIdentity;
  const auto [start, end] = getBounds(WORLD_RANK);

  if (WORLD_RANK >= NUM_PROC)
    goto finalise;

  // SEND THE DATA

  if (WORLD_RANK == 0) {
    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i);

      for (int j = qStart; j < qEnd; j++) {
        std::vector<double> matrixRow = matrix[j], identityRow = identity[j];
        MPI_Send(matrixRow.data(), matrixRow.size(), MPI_DOUBLE, i, 0,
                 MPI_COMM_WORLD);
        MPI_Send(identityRow.data(), identityRow.size(), MPI_DOUBLE, i, 0,
                 MPI_COMM_WORLD);
      }
    }

    localMatrix.resize(end - start);
    localIdentity.resize(end - start);
    for (int i = 0; i < localMatrix.size(); i++) {
      localMatrix[i] = matrix[start + i];
      localIdentity[i] = identity[start + i];
    }
  } else {
    localMatrix.resize(end - start);
    localIdentity.resize(end - start);
    for (int i = 0; i < localMatrix.size(); i++) {
      localMatrix[i].resize(N);
      MPI_Recv(localMatrix[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      localIdentity[i].resize(N);
      MPI_Recv(localIdentity[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }

  // perform gaussian elimination
  // todo: implement rule 1 of swapping rows
  for (int i = 0; i < N; i++) {
    const int RANK = findRank(i);

    // divide by Aii and subtract further rows. Assuming Aii != 0 for now.
    if (RANK == WORLD_RANK) {
      const int localIndex = i - start;

      if (localMatrix[localIndex][i] == 0) {
        // find a further row with non-zero pivot
        for (int j = localIndex + 1; j < localMatrix.size(); j++) {
          if (localMatrix[j][i] == 0)
            continue;
          std::swap(localMatrix[localIndex], localMatrix[j]);
          std::swap(localIdentity[localIndex], localIdentity[j]);
          break;
        }

        // search in other processes if required
        if (localMatrix[localIndex][i] == 0) {
          std::vector<double> matrixRow(N), identityRow(N);
          for (int j = RANK + 1; j < NUM_PROC; j++) {
            // send localMatrix[localIndex] and localIdentity[localIndex] to j
            std::vector<double> matrixRowToSend = localMatrix[localIndex],
                                identityRowToSend = localIdentity[localIndex];
            MPI_Send(matrixRowToSend.data(), N, MPI_DOUBLE, j, 1,
                     MPI_COMM_WORLD);
            MPI_Send(identityRowToSend.data(), N, MPI_DOUBLE, j, 1,
                     MPI_COMM_WORLD);
            // std::cout << "sending " << std::endl;
            // printVector(matrixRowToSend);
            // printVector(identityRowToSend);

            // read the status of the Recv
            MPI_Status status;
            MPI_Recv(matrixRow.data(), N, MPI_DOUBLE, j, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(identityRow.data(), N, MPI_DOUBLE, j, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            if (status.MPI_TAG != 1)
              continue;

            // set row as our new localIndex pivot row
            // std::cout << "setting row " << localIndex + start << " as pivot"
            //           << std::endl;
            // printVector(matrixRow);
            // printVector(identityRow);

            localMatrix[localIndex] = matrixRow;
            localIdentity[localIndex] = identityRow;
          }
        }
      }

      double pivot = localMatrix[localIndex][i];
      if (pivot == 0) {
        std::cerr << "matrix is not invertible" << std::endl;
        goto finalise;
      }

      for (int j = 0; j < N; j++) {
        localMatrix[localIndex][j] /= pivot;
        localIdentity[localIndex][j] /= pivot;
      }

      for (int row = localIndex + 1; row < localMatrix.size(); row++) {
        const double factor = localMatrix[row][i];
        for (int j = 0; j < N; j++) {
          localMatrix[row][j] -= factor * localMatrix[localIndex][j];
          localIdentity[row][j] -= factor * localIdentity[localIndex][j];
        }
      }

      // send the entire pivot rows to the other processes, so they can reduce
      // their values accordingly
      for (int j = RANK + 1; j < NUM_PROC; j++) {
        MPI_Send(localMatrix[localIndex].data(), N, MPI_DOUBLE, j, 0,
                 MPI_COMM_WORLD);
        MPI_Send(localIdentity[localIndex].data(), N, MPI_DOUBLE, j, 0,
                 MPI_COMM_WORLD);
      }
    } else if (WORLD_RANK > RANK) {
      MPI_Status status;
      std::vector<double> pivotRow(N), pivotIdentity(N);
      MPI_Probe(RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == 1) {
        MPI_Recv(pivotRow.data(), N, MPI_DOUBLE, RANK, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(pivotIdentity.data(), N, MPI_DOUBLE, RANK, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // check if we have a row with A_xi != 0, and return if present
        bool found = false;
        for (int row = 0; !found and row < localMatrix.size(); row++) {
          if (localMatrix[row][i] == 0)
            continue;
          std::vector<double> matrixRowToSend = localMatrix[row],
                              identityRowToSend = localIdentity[row];
          MPI_Send(matrixRowToSend.data(), N, MPI_DOUBLE, RANK, 1,
                   MPI_COMM_WORLD);
          MPI_Send(identityRowToSend.data(), N, MPI_DOUBLE, RANK, 1,
                   MPI_COMM_WORLD);
          localMatrix[row] = pivotRow;
          localIdentity[row] = pivotIdentity;
          found = true;
        }

        if (!found) {
          MPI_Send(pivotRow.data(), N, MPI_DOUBLE, RANK, 0, MPI_COMM_WORLD);
          MPI_Send(pivotIdentity.data(), N, MPI_DOUBLE, RANK, 0,
                   MPI_COMM_WORLD);
        }
      }

    reduceValues:
      MPI_Recv(pivotRow.data(), N, MPI_DOUBLE, RANK, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(pivotIdentity.data(), N, MPI_DOUBLE, RANK, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int row = 0; row < localMatrix.size(); row++) {
        const double factor = localMatrix[row][i];
        for (int j = 0; j < N; j++) {
          localMatrix[row][j] -= factor * pivotRow[j];
          localIdentity[row][j] -= factor * pivotIdentity[j];
        }
      }
    }
  }

  // back substitution phase
  for (int zCol = N - 1; zCol >= 0; zCol--) {
    const int RANK = findRank(zCol);

    if (RANK == WORLD_RANK) {
      const int localZCol = zCol - start;

      for (int row = localZCol - 1; row >= 0; row--) {
        const double factor = localMatrix[row][zCol];
        for (int col = 0; col < N; col++) {
          localMatrix[row][col] -= factor * localMatrix[localZCol][col];
          localIdentity[row][col] -= factor * localIdentity[localZCol][col];
        }
      }

      // send the entire pivot rows to the other processes
      for (int j = RANK - 1; j >= 0; j--) {
        MPI_Send(localMatrix[localZCol].data(), N, MPI_DOUBLE, j, 0,
                 MPI_COMM_WORLD);
        MPI_Send(localIdentity[localZCol].data(), N, MPI_DOUBLE, j, 0,
                 MPI_COMM_WORLD);
      }
    } else if (WORLD_RANK < RANK) {
      std::vector<double> pivotRow(N), pivotIdentity(N);
      MPI_Recv(pivotRow.data(), N, MPI_DOUBLE, RANK, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(pivotIdentity.data(), N, MPI_DOUBLE, RANK, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      for (int row = localMatrix.size() - 1; row >= 0; row--) {
        const double factor = localMatrix[row][zCol];
        for (int j = 0; j < N; j++) {
          localMatrix[row][j] -= factor * pivotRow[j];
          localIdentity[row][j] -= factor * pivotIdentity[j];
        }
      }
    }
  }

  // bring back the results to the root process
  if (WORLD_RANK != 0 && WORLD_RANK < NUM_PROC) {
    for (int i = 0; i < localMatrix.size(); i++) {
      MPI_Send(localMatrix[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(localIdentity[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  } else if (WORLD_RANK < NUM_PROC) {
    for (int i = 0; i < localMatrix.size(); i++) {
      matrix[start + i] = localMatrix[i];
      identity[start + i] = localIdentity[i];
    }

    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i);
      for (int j = qStart; j < qEnd; j++) {
        MPI_Recv(matrix[j].data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(identity[j].data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
    }

    std::cout << std::fixed << std::setprecision(2) << std::showpoint;

    for (int i = 0; i < N; i++) {
      // for (int j = 0; j < N; j++) {
      //   std::cout << matrix[i][j] << " ";
      // }
      for (int j = 0; j < N; j++)
        std::cout << identity[i][j] << " ";
      std::cout << std::endl;
    }
  }

finalise:
  MPI_Finalize();
  return 0;
}