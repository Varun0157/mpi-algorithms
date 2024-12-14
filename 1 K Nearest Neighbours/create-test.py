import random
import math
import heapq

# note: this script does not ensure distinct distances, and expects the output to be in order from least close to closest.

bounds = (-1e8, 1e8)
DECIMAL_PLACES = 2


def format_float(num):
    return f"{num:.2f}".rstrip("0").rstrip(".")


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_random_point():
    return (
        round(random.uniform(*bounds), DECIMAL_PLACES),
        round(random.uniform(*bounds), DECIMAL_PLACES),
    )


def generate_test_case(n, m, k):
    # Generate n points for set P
    P = [generate_random_point() for _ in range(n)]

    # Generate m query points for set Q with distinct distances
    Q = [generate_random_point() for _ in range(m)]

    # Write input to random-test.txt
    with open("test-inp.txt", "w") as f:
        f.write(f"{n} {m} {k}\n")
        for x, y in P:
            f.write(f"{x} {y}\n")
        for x, y in Q:
            f.write(f"{x} {y}\n")

    # Calculate and write output to random-test-opt.txt
    with open("test-opt.txt", "w") as f:
        for qx, qy in Q:
            distances = []
            for i, (px, py) in enumerate(P):
                dist = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
                heapq.heappush(
                    distances, (-dist, (px, py))
                )  # Use negative distance for max-heap
                if len(distances) > k:
                    heapq.heappop(distances)

            # Sort the k nearest points from least close to closest
            nearest = []
            while distances:
                _, (x, y) = heapq.heappop(distances)
                nearest.append((x, y))

            for (x, y) in nearest:
                f.write(f"{format_float(x)} {format_float(y)}\n")


# Example usage
n, m, k = 1000, 10000, 500  # You can change these values as needed
generate_test_case(n, m, k)
