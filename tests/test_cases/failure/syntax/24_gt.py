def floor(x: float) -> int:
    return int(x) if x - int(x) >= 0 else int(x) - 1


if __name__ == "__main__":
    import doctest

    doctest.testmod()
