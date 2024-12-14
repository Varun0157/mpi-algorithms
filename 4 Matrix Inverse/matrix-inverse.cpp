/*
Parallel matrix inverse using row reduction method and MPI
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <vector>

int finalise() {
  MPI_Finalize();
  return 0;
}

void storeMatrixAndIdentity(const char *filename, int &N,
                            std::vector<std::vector<double>> &matrix,
                            std::vector<std::vector<double>> &identi) {
  std::ifstream input(filename);
  if (!input) {
    std::cerr << "Error opening file!" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  input >> N;

  matrix.resize(N);
  identi.resize(N);
  for (int i = 0; i < N; i++) {
    matrix[i].resize(N);
    identi[i].resize(N);
    for (int j = 0; j < N; j++) {
      input >> matrix[i][j];
      identi[i][j] = (double)i == j;
    }
  }

  input.close();
}

std::pair<int, int> getBounds(const int rank, const int NUM_PROC, const int N) {
  const int baseItems = N / NUM_PROC;
  const int extraItems = N % NUM_PROC;
  const int start = baseItems * rank + std::min(rank, extraItems),
            end = start + baseItems + (rank < extraItems ? 1 : 0);

  return {start, end};
}

// inverse of getBounds
// given a row index, find the rank of the process that owns it
// I don't remember how I cooked this
int findRank(const int index, const int NUM_PROC, const int N) {
  const int baseItems = N / NUM_PROC;
  const int extraItems = N % NUM_PROC;

  int rank = (index < (baseItems + 1) * extraItems)
                 ? index / (baseItems + 1)
                 : (index - extraItems) / baseItems;

  return rank;
}

void sendRowsFromZero(const std::vector<std::vector<double>> &matrix,
                      const std::vector<std::vector<double>> &identi,
                      const int NUM_PROC, const int N) {
  for (int i = 1; i < NUM_PROC; i++) {
    const auto [qStart, qEnd] = getBounds(i, NUM_PROC, N);

    for (int j = qStart; j < qEnd; j++) {
      std::vector<double> matrixRow = matrix[j], identiRow = identi[j];
      MPI_Send(matrixRow.data(), matrixRow.size(), MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
      MPI_Send(identiRow.data(), identiRow.size(), MPI_DOUBLE, i, 0,
               MPI_COMM_WORLD);
    }
  }
}

int gaussianElimination(const int NUM_PROC, const int N, const int WORLD_RANK,
                        std::vector<std::vector<double>> &localMatrix,
                        std::vector<std::vector<double>> &localIdenti) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, N);

  for (int i = 0; i < N; i++) {
    const int RANK = findRank(i, NUM_PROC, N);

    if (RANK == WORLD_RANK) {
      // the index of the row in the local matrix
      const int localIndex = i - start;

      if (localMatrix[localIndex][i] == 0) {
        // find a further row in localMatrix with non-zero pivot
        for (int furtherRow = localIndex + 1; furtherRow < localMatrix.size();
             furtherRow++) {
          if (localMatrix[furtherRow][i] == 0)
            continue;
          std::swap(localMatrix[localIndex], localMatrix[furtherRow]);
          std::swap(localIdenti[localIndex], localIdenti[furtherRow]);
          break;
        }

        // search in other processes for a row with non zero pivot
        if (localMatrix[localIndex][i] == 0) {
          std::vector<double> matrixRow(N), identityRow(N);
          for (int furtherProc = RANK + 1; furtherProc < NUM_PROC;
               furtherProc++) {
            // send localMatrix[localIndex] and localIdentity[localIndex] to
            // furtherProc
            // tag = 1 means we are sending a row for gaussian elimination
            MPI_Send(localMatrix[localIndex].data(), N, MPI_DOUBLE, furtherProc,
                     1, MPI_COMM_WORLD);
            MPI_Send(localIdenti[localIndex].data(), N, MPI_DOUBLE, furtherProc,
                     1, MPI_COMM_WORLD);

            // read the status of the Recv
            // if it is 1, then we have found a row.
            MPI_Status status;
            MPI_Recv(matrixRow.data(), N, MPI_DOUBLE, furtherProc, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);
            MPI_Recv(identityRow.data(), N, MPI_DOUBLE, furtherProc,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG != 1)
              continue;

            localMatrix[localIndex] = matrixRow;
            localIdenti[localIndex] = identityRow;
          }
        }
      }

      double pivot = localMatrix[localIndex][i];
      if (pivot == 0) {
        std::cerr << "matrix is not invertible" << std::endl;
        return finalise();
      }

      for (int j = 0; j < N; j++) {
        localMatrix[localIndex][j] /= pivot;
        localIdenti[localIndex][j] /= pivot;
      }

      for (int row = localIndex + 1; row < localMatrix.size(); row++) {
        const double factor = localMatrix[row][i];
        for (int j = 0; j < N; j++) {
          localMatrix[row][j] -= factor * localMatrix[localIndex][j];
          localIdenti[row][j] -= factor * localIdenti[localIndex][j];
        }
      }

      // send the entire pivot rows to the other processes, so they can reduce
      // their values accordingly
      for (int furtherProc = RANK + 1; furtherProc < NUM_PROC; furtherProc++) {
        MPI_Send(localMatrix[localIndex].data(), N, MPI_DOUBLE, furtherProc, 0,
                 MPI_COMM_WORLD);
        MPI_Send(localIdenti[localIndex].data(), N, MPI_DOUBLE, furtherProc, 0,
                 MPI_COMM_WORLD);
      }
    } else if (WORLD_RANK > RANK) {
      MPI_Status status;
      std::vector<double> pivotRow(N), pivotIdentity(N);

      // just to check the tag
      MPI_Probe(RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == 1) {
        // tag = 1 means we are receiving a row for gaussian elimination

        MPI_Recv(pivotRow.data(), N, MPI_DOUBLE, RANK, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(pivotIdentity.data(), N, MPI_DOUBLE, RANK, 1, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // check if we have a row with A_xi != 0, and return with tag 1 if
        // present
        bool found = false;
        for (int row = 0; !found and row < localMatrix.size(); row++) {
          if (localMatrix[row][i] == 0)
            continue;
          MPI_Send(localMatrix[row].data(), N, MPI_DOUBLE, RANK, 1,
                   MPI_COMM_WORLD);
          MPI_Send(localIdenti[row].data(), N, MPI_DOUBLE, RANK, 1,
                   MPI_COMM_WORLD);
          localMatrix[row] = pivotRow;
          localIdenti[row] = pivotIdentity;
          found = true;
        }

        if (!found) {
          // return with tag 0 if we don't have a row with A_xi != 0
          MPI_Send(pivotRow.data(), N, MPI_DOUBLE, RANK, 0, MPI_COMM_WORLD);
          MPI_Send(pivotIdentity.data(), N, MPI_DOUBLE, RANK, 0,
                   MPI_COMM_WORLD);
        }
      }

      // reduce values based on the pivot row
      MPI_Recv(pivotRow.data(), N, MPI_DOUBLE, RANK, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(pivotIdentity.data(), N, MPI_DOUBLE, RANK, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int row = 0; row < localMatrix.size(); row++) {
        const double factor = localMatrix[row][i];
        for (int j = 0; j < N; j++) {
          localMatrix[row][j] -= factor * pivotRow[j];
          localIdenti[row][j] -= factor * pivotIdentity[j];
        }
      }
    }
  }

  return 0;
}

void backSubstitution(const int NUM_PROC, const int N, const int WORLD_RANK,
                      std::vector<std::vector<double>> &localMatrix,
                      std::vector<std::vector<double>> &localIdenti) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, N);

  for (int zCol = N - 1; zCol >= 0; zCol--) {
    const int RANK = findRank(zCol, NUM_PROC, N);

    if (RANK == WORLD_RANK) {
      const int localZCol = zCol - start;

      for (int row = localZCol - 1; row >= 0; row--) {
        const double factor = localMatrix[row][zCol];
        for (int col = 0; col < N; col++) {
          localMatrix[row][col] -= factor * localMatrix[localZCol][col];
          localIdenti[row][col] -= factor * localIdenti[localZCol][col];
        }
      }

      // send the entire pivot rows to the other processes
      for (int j = RANK - 1; j >= 0; j--) {
        MPI_Send(localMatrix[localZCol].data(), N, MPI_DOUBLE, j, 0,
                 MPI_COMM_WORLD);
        MPI_Send(localIdenti[localZCol].data(), N, MPI_DOUBLE, j, 0,
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
          localIdenti[row][j] -= factor * pivotIdentity[j];
        }
      }
    }
  }
}

void partitionData(const int N, const int NUM_PROC, const int WORLD_RANK,
                   std::vector<std::vector<double>> &matrix,
                   std::vector<std::vector<double>> &identi,
                   std::vector<std::vector<double>> &localMatrix,
                   std::vector<std::vector<double>> &localIdenti) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, N);

  // send the required rows to the other processes
  if (WORLD_RANK == 0) {
    sendRowsFromZero(matrix, identi, NUM_PROC, N);

    localMatrix.resize(end - start);
    localIdenti.resize(end - start);
    for (int i = 0; i < localMatrix.size(); i++) {
      localMatrix[i] = matrix[start + i];
      localIdenti[i] = identi[start + i];
    }

    matrix.clear();
    identi.clear();
  } else {
    localMatrix.resize(end - start);
    localIdenti.resize(end - start);
    for (int i = 0; i < localMatrix.size(); i++) {
      localMatrix[i].resize(N);
      MPI_Recv(localMatrix[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      localIdenti[i].resize(N);
      MPI_Recv(localIdenti[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
}

std::chrono::time_point<std::chrono::system_clock>
accumulateAndPrint(const int NUM_PROC, const int N, const int WORLD_RANK,
                   const int WORLD_SIZE,
                   std::vector<std::vector<double>> &matrix,
                   std::vector<std::vector<double>> &identi,
                   std::vector<std::vector<double>> &localMatrix,
                   std::vector<std::vector<double>> &localIdenti) {
  const auto [start, end] = getBounds(WORLD_RANK, NUM_PROC, N);

  // bring back the results to the root process
  // only the identity really matters
  if (WORLD_RANK != 0) {
    for (int i = 0; i < localMatrix.size(); i++) {
      // MPI_Send(localMatrix[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(localIdenti[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  } else {
    // matrix.resize(N);
    identi.resize(N);

    for (int i = 0; i < localMatrix.size(); i++) {
      // matrix[start + i] = localMatrix[i];
      identi[start + i] = localIdenti[i];
    }

    for (int i = 1; i < NUM_PROC; i++) {
      const auto [qStart, qEnd] = getBounds(i, NUM_PROC, N);
      for (int j = qStart; j < qEnd; j++) {
        // matrix[j].resize(N);
        // MPI_Recv(matrix[j].data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
        //          MPI_STATUS_IGNORE);
        identi[j].resize(N);
        MPI_Recv(identi[j].data(), N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
    }
  }

  return std::chrono::system_clock::now();
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
  std::vector<std::vector<double>> matrix, identi;
  if (WORLD_RANK == 0) {
    storeMatrixAndIdentity(argv[1], N, matrix, identi);
  }

  std::chrono::time_point<std::chrono::system_clock> beginTime =
      std::chrono::system_clock::now();

  // broadcast the size of the matrix to all processes
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int NUM_PROC = std::min(WORLD_SIZE, N);

  if (WORLD_RANK >= NUM_PROC)
    return finalise();

  std::vector<std::vector<double>> localMatrix, localIdenti;
  partitionData(N, NUM_PROC, WORLD_RANK, matrix, identi, localMatrix,
                localIdenti);
  gaussianElimination(NUM_PROC, N, WORLD_RANK, localMatrix, localIdenti);
  backSubstitution(NUM_PROC, N, WORLD_RANK, localMatrix, localIdenti);

  auto endTime = accumulateAndPrint(NUM_PROC, N, WORLD_RANK, WORLD_SIZE, matrix,
                                    identi, localMatrix, localIdenti);

  if (WORLD_RANK != 0)
    return finalise();

  std::ofstream output(std::to_string(N) + "_time.txt", std::ios::app);
  output << WORLD_SIZE << ":"
         << std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                 beginTime)
                .count()
         << std::endl;

  std::cout << std::fixed << std::setprecision(2) << std::showpoint;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      std::cout << identi[i][j] << " ";
    std::cout << std::endl;
  }

  return finalise();
}