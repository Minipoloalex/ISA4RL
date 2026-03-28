import string
import random

def generate_random_string(sz: int) -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(sz))
