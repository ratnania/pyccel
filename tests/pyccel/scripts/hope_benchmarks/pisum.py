
def pisum():
    for j in range(1, 501):
        sum = 0.0
        for k in range(1, 10001):
            sum += 1.0/(k*k)
    return sum

print(pisum())
