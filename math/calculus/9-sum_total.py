def summation_i_squared(n):
    # Check if n is a valid integer and greater than or equal to 1
    if not isinstance(n, int) or n < 1:
        return None

    # Base case: When n is 1, return 1 (1^2)
    if n == 1:
        return 1

    # Recursive case: Calculate n^2 and add it to the sum of squares of previous numbers
    return n**2 + summation_i_squared(n - 1)
