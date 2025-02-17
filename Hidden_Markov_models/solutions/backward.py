def backward_computation( A, B, observations):
    T = len(observations)
    N = A.shape[0]
    beta = np.zeros((T, N))
 
    # Initialization
    beta[T - 1] = np.ones((N))
 
    # Recursion
    for t in range(T - 2, -1, -1):
        for j in range(N):
            product = beta[t+1, :] * A[j, :] * B[:, observations[t+1]]
            beta[t, j] = np.sum(product)
 
    return beta