from __future__ import annotations

from collections.abc import Callable


def trapezoidal_area(
    fnc: Callable[[float], float],
    x_start: float,
    x_end: float,
    steps: int = 100,
) -> float:
    x1 = x_start
    fx1 = fnc(x_start)
    area = 0.0
    for _ in range(steps):
        # Approximates small segments of curve as linear and solve
        # for trapezoidal area
        x2 = (x_end - x_start) / steps + x1
        fx2 = fnc(x2)
        area += abs(fx2 + fx1) * (x2 - x1) / 2
        # Increment step
        x1 = x2
        fx1 = fx2
    return area


if __name__ == "__main__":

    def f(x):
        return x**3 + x**2

    print("f(x) = x^3 + x^2")
    print("The area between the curve, x = -5, x = 5 and the x axis is:")
    i = 10
    while i <= 100000:
        print(f"with {i} steps: {trapezoidal_area(f, -5, 5, i)}")
        i *= 10
