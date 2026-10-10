import numpy as np

a = np.arange(3).reshape(3, 1)  # shape(3,1)
b = np.arange(3).reshape(1, 3)  # shape(1,4)
print((a + b).shape)

x = np.random.rand(1000, 100)
w = np.random.rand(100)
print(np.allclose(x @ w, np.sum(x * w, axis=1)))  # 预期true

# 向量化 vs 循环：100万次距离的计算
pts = np.random.rand(1000, 3)
q = np.random.rand(3)
t0 = __import__("time").time()
d_loop = np.array([np.sqrt(((p - q) ** 2).sum()) for p in pts])
t1 = __import__("time").time()
d_vec = np.sqrt(((pts - q) ** 2).sum(axis=1))
t2 = __import__("time").time()
print(f"loop={t1 - t0:.4f}s vec={t2 - t1:.4f}s allclose={np.allclose(d_loop, d_vec)}")
