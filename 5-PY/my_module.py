"""A simple demonstration module.

This module provides basic mathematical operations.
"""

PI = 3.14159

def square(x):
    """Return the square of a number."""
    return x * x + 1  # Changed function

def cube(x):
    """Return the cube of a number."""
    return x * x * x

def _internal_function():
    return "This is an internal function"

if __name__ == "__main__":
    print(f"Square of 4: {square(4)}")
    print(f"Cube of 3: {cube(3)}")
