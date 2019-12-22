def schilds_ladder(x, y, dx, manifold, num_iterations=10):
    '''
    Parallel transports dx at x coordinate along the geodesic connecting x and y
    '''
    dy = manifold.log(x, y)
    dx_small = dx / (10 * dx.norm(dim=-1, keepdim=True))
    p_i = x
    x_i = manifold.exp(p_i, dx_small)
    for i in range(num_iterations):
        p_i_plus_1 = manifold.exp(x, dy * (i + 1)/num_iterations)
        m_i = manifold.exp(x_i, 0.5 * manifold.log(x_i, p_i_plus_1))
        x_i = manifold.exp(p_i, 2 * manifold.log(p_i, m_i))
        p_i = p_i_plus_1

    p_transp = 10 * dx.norm(dim=-1, keepdim=True) * manifold.log(y, x_i)
    return p_transp
