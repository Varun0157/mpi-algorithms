import random
import math
import heapq

# note: this script does not ensure distinct distances, and expects the output to be in order from least close to closest.

bounds = (-1e4, 1e4)


def format_float(num):
    return f"{num:.2f}".rstrip("0").rstrip(".")


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def generate_unique_point(existing_points, max_attempts=100):
    for _ in range(max_attempts):
        candidate = (
            round(random.uniform(*bounds), 2),
            round(random.uniform(*bounds), 2),
        )
        if candidate not in existing_points:
            return candidate
    raise ValueError("Could not generate a unique point after maximum attempts")


def generate_distinct_point(P, existing_Q, max_attempts=100):
    for _ in range(max_attempts):
        candidate = (
            round(random.uniform(*bounds), 2),
            round(random.uniform(*bounds), 2),
        )
        distances = set()
        is_distinct = True

        for p in P:
            dist = distance(candidate, p)
            if dist in distances:
                is_distinct = False
                break
            distances.add(dist)

        for q in existing_Q:
            for p in P:
                if abs(distance(candidate, p) - distance(q, p)) < 1e-5:
                    is_distinct = False
                    break
            if not is_distinct:
                break

        if is_distinct:
            return candidate

    raise ValueError(
        "Could not generate a point with distinct distances after maximum attempts"
    )


def generate_test_case(n, m, k):
    # Generate n points for set P
    P = []
    for _ in range(n):
        new_point = generate_unique_point(P)
        P.append(new_point)

    # Generate m query points for set Q with distinct distances
    Q = []
    for _ in range(m):
        new_point = generate_distinct_point(P, Q)
        Q.append(new_point)

    # Write input to random-test.txt
    with open("random-test.txt", "w") as f:
        f.write(f"{n} {m} {k}\n")
        for x, y in P:
            f.write(f"{x:.2f} {y:.2f}\n")
        for x, y in Q:
            f.write(f"{x:.2f} {y:.2f}\n")

    # Calculate and write output to random-test-opt.txt
    with open("random-test-opt.txt", "w") as f:
        for qx, qy in Q:
            distances = []
            for i, (px, py) in enumerate(P):
                dist = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
                heapq.heappush(
                    distances, (-dist, i)
                )  # Use negative distance for max-heap
                if len(distances) > k:
                    heapq.heappop(distances)

            # Sort the k nearest points from least close to closest
            nearest = []
            while distances:
                _, i = heapq.heappop(distances)
                nearest.append(i)

            for i in nearest:
                x, y = P[i]
                f.write(f"{format_float(x)} {format_float(y)}\n")


# Example usage
n, m, k = 100, 50, 45  # You can change these values as needed
generate_test_case(n, m, k)
