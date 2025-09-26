import jax

key = jax.random.PRNGKey(0)

print("Initial key:", key)
key, k1 = jax.random.split(key)
print("After first split k1:", k1)
print("After first split key:", key)
key, k2 = jax.random.split(key)
print("After second split k2:", k2)
print("After second split key:", key)


x = jax.random.normal(k1, (3,))
y = jax.random.normal(k2, (3,))
z = jax.random.normal(k1, (4,))

print("x:", x)
print("y:", y)
print("z:", z)