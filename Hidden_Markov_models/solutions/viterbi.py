def viterbi(A, B, pi, observations):
    """
    Returns the optimal state sequence and the probability associated
    """

    N = A.shape[0]    # Number of states
    T = len(observations)  # Length of observation sequence

    v = np.zeros((N, T))
    bt = np.zeros((N, T-1))
    Q_opt = np.zeros(T)

    ### Initialization
    v[:, 0] = pi * B[:, observations[0]]

    ### Recursion
    for n in range(1, T):
        for i in range(N):
            temp_product = A[:, i] * v[:, n-1]* B[i, observations[n]]
            v[i, n] = np.max(temp_product)
            bt[i, n-1] = np.argmax(temp_product)

    ###Termination    
    # Backtracking
    Q_opt[-1] = np.argmax(v[:, -1])
    for n in range(T-2, -1, -1):
        Q_opt[n] = bt[int(Q_opt[n+1]), n]
    
    # Best probability
    best_prob = np.max(v[:, T-1])
    #print(v)

    return Q_opt, best_prob