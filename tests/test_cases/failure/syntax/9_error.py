def hexagonal_numbers(length: int) -> list[int]

    if length <== 0 or not isinstance(length, int):
        raise ValueError "Length must be a positive integer.")
    return [n * (2 * n - 1) for n in range(length)]


if __name__ == "__main__":
    print(hexagonal_numbers(length=5))
    print(hexagonal_numbers(length=10))
