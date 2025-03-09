from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Node:
    data: float
    left: Node | None = None
    right: Node | None = None

    def __iter__(self) -> Iterator[float]:
        if self.left:
            yield from self.left
        yield self.data
        if self.right:
            yield from self.right

    @property
    def is_sorted(self) -> bool:
        if self.left and (self.data < self.left.data or not self.left.is_sorted):
            return False
        return not (
            self.right and (self.data > self.right.data or not self.right.is_sorted)
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    tree = Node(data=2.1, left=Node(data=2.0), right=Node(data=2.2))
    print(f"Tree {list(tree)} is sorted: {tree.is_sorted = }.")
    assert tree.right
    tree.right.data = 2.0
    print(f"Tree {list(tree)} is sorted: {tree.is_sorted = }.")
    tree.right.data = 2.1
    print(f"Tree {list(tree)} is sorted: {tree.is_sorted = }.")
