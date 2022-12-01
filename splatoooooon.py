import taichi as ti

ti.init(arch=ti.gpu)

# a = float('inf')
# b = float('-inf')

mouse_circle = ti.Vector.field(2, dtype=float, shape=(1, ))

def main():
    print(
        "splatoooooon!"
    )

    res = (1080, 720)
    window = ti.ui.Window("splatoooooon", res, vsync=True)
    canvas = window.get_canvas()
    global radius
    radius = 0.003

    # GGUI
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key in [ti.ui.ESCAPE]:
                break
        mouse = window.get_cursor_pos()
        mouse_circle[0] = ti.Vector([mouse[0], mouse[1]])
        canvas.circles(mouse_circle, color=(0.2, 0.4, 0.6), radius=0.01)
        if window.is_pressed(ti.ui.LMB):
            canvas.circles(mouse_circle, color=(0.8, 0.1, 0.1), radius=0.05)
        canvas.set_background_color((0.5, 0.5, 0.5))
        window.show()

if __name__ == '__main__':
    main()