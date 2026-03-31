from micrograd.nn import MLP

def is_prime(n):
    n = int(n)
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

xs = [
    [2.0, 3.0, 5.0, 7.0],    # sum=17  prime
    [2.0, 4.0, 5.0, 7.0],    # sum=18  not
    [11.0, 13.0, 17.0, 19.0],# sum=60  not
    [3.0, 6.0, 7.0, 11.0],   # sum=27  not
    [1.0, 2.0, 3.0, 5.0],    # sum=11  prime
    [4.0, 6.0, 8.0, 9.0],    # sum=27  not
    [5.0, 7.0, 11.0, 13.0],  # sum=36  not
    [1.0, 3.0, 5.0, 2.0],    # sum=11  prime
    [1.0, 1.0, 1.0, 4.0],    # sum=7   prime
    [2.0, 3.0, 4.0, 5.0],    # sum=14  not
    [10.0, 10.0, 10.0, 1.0], # sum=31  prime
    [8.0, 9.0, 10.0, 11.0],  # sum=38  not
    [1.0, 1.0, 1.0, 2.0],    # sum=5   prime
    [12.0, 13.0, 17.0, 19.0],# sum=61  prime
    [3.0, 3.0, 3.0, 3.0],    # sum=12  not
    [1.0, 2.0, 4.0, 6.0],    # sum=13  prime
    [5.0, 5.0, 5.0, 5.0],    # sum=20  not
    [10.0, 10.0, 10.0, 7.0], # sum=37  prime
    [6.0, 6.0, 6.0, 6.0],    # sum=24  not
    [1.0, 1.0, 10.0, 1.0],   # sum=13  prime
]

# 1.0 if sum is prime, -1.0 if not (tanh-friendly targets)
ys = [1.0 if is_prime(sum(x)) else -1.0 for x in xs]

n = MLP(4, [4, 4, 1])

for k in range(3000):
    ypreds = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypreds))

    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    for p in n.parameters():
        p.data -= 0.00005 * p.grad

    if k % 50 == 0:
        print(f"Step {k}: Loss = {loss.data:.6f}")

print(f"\nFinal Loss = {loss.data:.6f}")

test_xs = [
    [3.0, 3.0, 3.0, 2.0],   # sum=11  prime   -> expect 1
    [2.0, 2.0, 2.0, 2.0],   # sum=8   not     -> expect 0
    [1.0, 1.0, 1.0, 1.0],   # sum=4   not     -> expect 0
    [1.0, 2.0, 3.0, 7.0],   # sum=13  prime   -> expect 1
    [10.0, 10.0, 10.0, 3.0],# sum=33  not     -> expect 0
    [20.0, 20.0, 10.0, 9.0],# sum=59  prime   -> expect 1
]

print("\n--- Predictions ---")
for x in test_xs:
    s = int(sum(x))
    prediction = n(x)
    output = 1 if prediction.data > 0 else 0
    label = "prime" if is_prime(s) else "not prime"
    print(f"Input: {x} | Sum: {s} ({label}) | Raw: {prediction.data:.4f} | Output: {output}")
