import numpy as np
from tqdm import trange

def solve_laplace(x_grid, y_grid, boundary_condition, tolerance=1e-4, max_iterations=10000):
    # Laplace equation on a rectangular domain with BC
    
    n_x = len(x_grid)
    n_y = len(y_grid)
    
    u_sol = np.zeros((n_x, n_y))
    
    # Apply boundary conditions
    for ix in range(n_x):
        for iy in range(n_y):
            if ix == 0 or ix == n_x-1 or iy == 0 or iy == n_y-1:
                u_sol[ix, iy] = boundary_condition(x_grid[ix], y_grid[iy])
                
    for n_iter in trange(max_iterations):
        u_old = u_sol.copy()
        for ix in range(1, n_x-1):
            for iy in range(1, n_y-1):
                u_sol[ix, iy] = (u_sol[ix-1,iy]+u_sol[ix+1,iy]+u_sol[ix,iy-1]+u_sol[ix,iy+1])/4
        
        diff=np.linalg.norm(u_sol-u_old)
        if diff < tolerance:
            break
            
        if n_iter == max_iterations-1:
            print("maximum number of iterations reached")

    return u_sol