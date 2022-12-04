import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu, debug=True)

# constant
Resolution = (1080, 720)
steps = 25
dt = 2e-4
GRAVITY = -9.8
PI = 3.1415
E = 800

# grid properties
n_grids = 2**6
g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
g_m = ti.field(float, (n_grids, n_grids, n_grids))

dx = 1 / n_grids

# particle properties
n_particles = 2**12
p_rho = 1.2
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
p_x = ti.Vector.field(3, float, n_particles)
p_x_origin = ti.Vector.field(3, float, shape=())
p_x_origin[None] = (0.5, 0.5, 0.8)
p_x_xz = ti.Vector.field(2, float, n_particles)  #particle position without height(y_value)
p_v = ti.Vector.field(3, float, n_particles)
p_v0 = 5.0
p_v0_direction = ti.Vector.field(3, float, shape=())
p_v0_direction[None] = (0, 0, -1)
p_colors = ti.Vector.field(4, float, n_particles)
p_C = ti.Matrix.field(3, 3, float, n_particles)  
p_Jp = ti.field(float, n_particles)  



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

        stress = -dt * 4 * E * p_vol * (p_Jp[p] - 1) / dx**2  ##internal force [MSLMPM]
        affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]]) + p_mass * p_C[p]  #[MSLMPM]

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
    for I in ti.grouped(g_m):
        if g_m[I] > 0:
            g_v[I] /= g_m[I]
        g_v[I] += dt * ti.Vector([0, GRAVITY, 0])
        #print(g_v[I])
        # BC
        cond = (I < 3) & (g_v[I] < 0) | (I > n_grids - 3) & (g_v[I] > 0)
        g_v[I] = 0 if cond else g_v[I]
    # G2P
    for p in p_x:
        Xp = p_x[p] / dx
        base = int(Xp - 0.5)  #floor(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_p_v = ti.zero(p_v[p])
        new_p_C = ti.zero(p_C[p])
        for offset in ti.static(ti.ndrange(3,3,3)):
            dpos = (offset - fx) * dx
            weight = 1.0
            # calculate x y z component for every neighbor grid 
            for i in ti.static(range(3)):
                weight *= w[offset[i]][i]  #???WHY
            grid_v = g_v[base + offset]
            new_p_v += weight * grid_v
            new_p_C += 4 * weight * grid_v.outer_product(dpos) / dx**2
        p_v[p] = new_p_v
        p_x[p] += dt * p_v[p]
        p_Jp[p] *= 1 + dt * new_p_C.trace() #particle volume (MSLMPM)
        p_C[p] = new_p_C
        p_x_xz[p] = (p_x[p][0], p_x[p][2])



@ti.kernel
def init_particle():
    for n in range(n_particles):
        # shape of particle cluster
        R = 0.02
        p_colors[n] = (0.1, 0.6, 0.9, 1.0)
        theta = ti.random() * 2 * PI
        phi = ti.random() * PI
        sin_theta, cos_theta = tm.sin(theta), tm.cos(theta)
        sin_phi, cos_phi = tm.sin(phi), tm.cos(phi)

        p_v0_direction[None] = tm.normalize(p_v0_direction[None])
        p_v[n] = p_v0_direction[None] * p_v0

        p_x[n] = [R * sin_phi * cos_theta, R * sin_phi * sin_theta, 3 * R * cos_phi]
        alpha = tm.acos(tm.dot(p_v0_direction[None], ti.Vector([0, 0, -1])))
        p_x[n] = ti.Matrix([
            [tm.cos(alpha), 0, tm.sin(alpha)],
            [0, 1, 0], 
            [-tm.sin(alpha), 0, tm.cos(alpha)]]) @ p_x[n]
        p_x[n] += p_x_origin[None]

        p_Jp[n] = 1
        p_C[n] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])



# init_render
window = ti.ui.Window("splatoooooon 3D", Resolution, vsync=True)
canvas = window.get_canvas()
gui = window.get_gui()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
def init_render():
    camera.position(0.5, 1.0, 1.95)
    camera.lookat(0.5, 0.3, 0.5)
    camera.fov(55)


# init_options
ggui_particales_radius = 0.01
paused = False
def show_options():
    global ggui_particales_radius
    global paused

    with gui.sub_window("Options",  0.05, 0.45, 0.2, 0.4) as w:
        if w.button("restart"):
            init_particle()
            paused = True
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True



mouse_position = ti.Vector.field(2, float, shape=(1, ))
p_x_origin_xz = ti.Vector.field(2, float, shape=(1, ))
p_x_origin_xz[0] = ti.Vector([p_x_origin[None][0], p_x_origin[None][2]])
line_xz = ti.Vector.field(2, dtype=float, shape = 2)
line_xz[0] = p_x_origin_xz[0]
def draw():
    # 3D scene
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0.5, 0.5, 0.5))

    scene.particles(p_x, per_vertex_color = p_colors, radius = ggui_particales_radius)
    #scene.particles(p_x_origin, per_vertex_color = (1, 1, 1), radius = 0.01)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

    # 2D canvas
    canvas.circles(p_x_xz, color=(0.1, 0.1, 0.1), radius = ggui_particales_radius)
    canvas.circles(p_x_origin_xz, color=(1, 1, 1), radius = ggui_particales_radius)

    mouse = window.get_cursor_pos()
    mouse_position[0] = ti.Vector([mouse[0], mouse[1]])
    line_xz[1] = mouse_position[0]
    canvas.circles(mouse_position, color=(0.2, 0.4, 0.6), radius=0.01)
    canvas.lines(line_xz, color = (0.2, 0.2, 0.2), width = 0.01)
    if window.is_pressed(ti.ui.LMB):
        canvas.circles(mouse_position, color=(0.8, 0.1, 0.1), radius=0.05)
        p_v0_direction[None] = (mouse_position[0][0] - p_x_origin_xz[0][0], 0, mouse_position[0][1] - p_x_origin_xz[0][1])
        init_particle()

    



def main():
    print(
        "Press Left-key to shoot, hold Right-key to rotate. \n",
        "The gray particles in the background is the top-down view"
        )
    init_particle()
    init_render()
    while window.running:
        if not paused:
            for _ in range(steps):
                substep()

        draw()
        show_options()

        window.show()
    


if __name__ == '__main__':
    main()