# Define the logical operations
def OR(a, b):
    return a or b

def NOT(a):
    return not a

def NOR(a, b):
    return not (a or b)

def XOR(a, b):
    return a != b

# OR operation
print("\nOR")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {OR(*x)}")

# NOT operation
print("\nNOT")
for x in [0, 1]:
    print(f"{x} -> {NOT(x)}")

# NOR operation
print("\nNOR")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {NOR(*x)}")

# XOR operation
print("\nXOR")
for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f"{x} -> {XOR(*x)}")
