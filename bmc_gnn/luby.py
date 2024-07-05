import math

def is_pow_2(n: int) -> bool:
    """ Test if n is some 2**k. """
    return ((n) & (n - 1)) == 0 and n != 0

def max_pow_2_in(n: int) -> int:
    """ Return the highest power of 2 less than or equal to than n. """
    return 2 ** int(math.log2(n))

def luby(i: int) -> int:
    """ Return the i-th luby number. """

    if is_pow_2(i + 1):
        return (i + 1) >> 1

    return luby((i + 1) - max_pow_2_in(i))
