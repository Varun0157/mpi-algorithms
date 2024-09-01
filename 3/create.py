import random

def format_float(num) -> str:
    return f"{num:.2f}".rstrip("0").rstrip(".")

def generate_test_case(n):
    # Generate n random floating point numbers between 0 and 10 with 2 decimal places
    numbers = [round(random.uniform(0, 10), 2) for _ in range(n)]

    # Calculate prefix sums
    prefix_sums = [round(sum(numbers[: i + 1]), 2) for i in range(n)]

    return numbers, prefix_sums


def write_files(n):
    numbers, prefix_sums = generate_test_case(n)

    # Write input file
    with open("random.txt", "w") as f:
        f.write(f"{n}\n")
        f.write(" ".join(map(format_float, numbers)))
        f.write(" \n")

    # Write output file
    with open("random-opt.txt", "w") as f:
        f.write(" ".join(map(format_float, prefix_sums)))
        f.write(" \n")


# Set n manually here
n = 1750  # works for 1750 nums

write_files(n)
print(f"Files 'random.txt' and 'random-opt.txt' have been created with {n} numbers.")
