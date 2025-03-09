from __future__ import annotations


def encode(plain: str) -> list[int]:
    """
    >>> encode("myname")
    [13, 25, 14, 1, 13, 5]
    """
    return [ord(elem) - 96 for elem in plain if elem != " "]


def decode(encoded: list[int]) -> str:
    """
    >>> decode([13, 25, 14, 1, 13, 5])
    'myname'
    """
    return "".join(chr(elem + 97) for elem in encoded)


def main() -> None:
    encoded = encode(input("-> ").strip().lower())
    print("Encoded: ", encoded)
    print("Decoded:", decode(encoded))


if __name__ == "__main__":
    main()
