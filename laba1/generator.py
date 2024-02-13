import random, sys

n = int(sys.argv[1])

with open("test.txt", 'w') as out:
    out.write(str(n) + "\n")
    for i in range(n+1):
        out.write(" ".join(map(str, [random.uniform(-10, 10) for x in range(n)])) + "\n")
