def forward(A, B, pi, observations):
    """
    Returns the likelihood probability
    """

    T = len(observations)
    N = pi.shape[0]
    
    alpha = np.zeros((N, T))
    
    # Initialization
    
    alpha[:, 0] = pi * B[:,observations[0]]
    
    # Recursion
    for t in range(1, T):
        for j in range(N):
            for i in range(N):
                alpha[j, t] += alpha[i, t-1] * A[i, j] * B[j, observations[t]]
    
    print("Complete alpha matrix:\n",alpha)
    
    return np.sum(alpha[:, T-1])