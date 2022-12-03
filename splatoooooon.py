import taichi as ti

ti.init(arch=ti.gpu)

# constant
steps = 25
dt = 2e-4
GRAVITY = -9.8
gravity = ti.Vector.field(3, dtype=float, shape=())
gravity[None] = [0, GRAVITY, 0]
E = 1000

# grid properties
n_grids = 64
g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
g_m = ti.field(float, (n_grids, n_grids, n_grids))

dx = 1 / n_grids

# particle properties
n_particles = 2**14
p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
p_x = ti.Vector.field(3, float, n_particles)
p_v = ti.Vector.field(3, float, n_particles)
p_acc = ti.Vector.field(3, float, shape=())
p_colors = ti.Vector.field(4, float, n_particles)
p_C = ti.Matrix.field(3, 3, float, n_particles)
p_J = ti.field(float, n_particles)
#p_used = ti.field(int, n_particles)

@ti.kernel
def substep():
    # init grid
    for i, j, k in g_m:
        g_v[i, j, k] = [0, 0, 0]
        g_m[i, j, k] = 0
    # P2G
    for p in p_x:
        Xp = p_x[p] / dx
        base = int(Xp - 0.5)  #floor(Xp - 0.5)
        fx = Xp - base
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (p_J[p] - 1) / dx**2  #internal force [MSLMPM]
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * p_C[p]  #[MSLMPM]
        # loop over 3*3*3 grid node neighbor
        for offset in ti.static(ti.ndrange(3,3,3)):
            dpos = (offset - fx) * dx
            weight = 1.0
            # calculate x y z component for every neighbor grid 
            for i in ti.static(range(3)):
                weight *= w[offset[i]][i]  #???WHY
                g_v[base + offset] += weight * (p_mass * p_v[p] + affine @ dpos)  #Grid momentum / mass [MSLMPM]
                g_m[base + offset] += weight * p_mass   #Grid mass
    # Grid operations
    for i, j, k in g_m:


    
def main():
    substep()

if __name__ == '__main__':
    main()