from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import math
# import numpy as np

name = 'ball_glut'


v0 = (+1, -1, -1)
v1 = (+1, +1, -1)
v2 = (-1, +1, -1)
v3 = (-1, -1, -1)
v4 = (+1, -1, +1)
v5 = (+1, +1, +1)
v6 = (-1, +1, +1)
v7 = (-1, -1, +1)

right = (1, 0, 0)
left = (-1, 0, 0)
top = (0, 1, 0)
bottom = (0, -1, 0)
front = (0, 0, 1)
back = (0, 0, -1)

vertices = [
    v0, v1, v5,
    v0, v5, v4,
    v0, v3, v2,
    v0, v2, v1,
    v3, v7, v6,
    v3, v6, v2,
    v4, v5, v6,
    v4, v6, v7,
    v1, v2, v6,
    v1, v6, v5,
    v0, v4, v7,
    v0, v7, v3,
]

normals = [
    right, right, right,
    right, right, right,
    back, back, back,
    back, back, back,
    left, left, left,
    left, left, left,
    front, front, front,
    front, front, front,
    top, top, top,
    top, top, top,
    bottom, bottom, bottom,
    bottom, bottom, bottom,
]

eye = [2, 2, 10]
target = [0, 0, 0]
up = [0, 1, 0]
fov_y = None
aspect = None
near = None
far = None
previous_point = None
window = None
button_down = None


def mouse_func(button, state, x, y):
    global previous_point, button_down
    print(button_down, state, x, y)
    previous_point = (x * 2 / window[0], -y * 2 / window[1])
    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            button_down = 'left'
        elif state == GLUT_UP:
            button_down = None
    elif button == GLUT_RIGHT_BUTTON:
        if state == GLUT_DOWN:
            button_down = 'right'
        elif state == GLUT_UP:
            button_down = None


def motion_func(x, y):
    # this function modeled after modeler.PerspectiveCamera.orbit() function written by 'ags' here:
    # http://www.cs.cornell.edu/courses/cs4620/2008fa/asgn/model/model-fmwk.zip
    global previous_point, eye
    x *= 2 / window[0]
    y *= -2 / window[1]
    if button_down == 'left':
        mouse_delta = [x - previous_point[0], y - previous_point[1]]
        neg_gaze = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
        dist = sum([a**2 for a in neg_gaze]) ** (1/2)
        neg_gaze = [a / dist for a in neg_gaze]
        azimuth = math.atan2(neg_gaze[0], neg_gaze[2])
        elevation = math.atan2(neg_gaze[1], (neg_gaze[0]**2 + neg_gaze[2]**2)**(1/2))
        azimuth = (azimuth - mouse_delta[0]) % (2 * math.pi)
        elevation = max(-math.pi * .495, min(math.pi * .495, elevation - mouse_delta[1]))
        neg_gaze[0] = math.sin(azimuth) * math.cos(elevation)
        neg_gaze[1] = math.sin(elevation)
        neg_gaze[2] = math.cos(azimuth) * math.cos(elevation)
        mag = sum([a**2 for a in neg_gaze]) ** (1/2)
        neg_gaze = [a / mag * dist for a in neg_gaze]
        new_eye = [a + b for a, b in zip(target, neg_gaze)]
        eye = new_eye
        glutPostRedisplay()
    elif button_down == 'right':
        mouse_delta_y = y - previous_point[1]
        neg_gaze = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]
        dist = sum([a**2 for a in neg_gaze]) ** (1/2)
        new_dist = dist - 5 * mouse_delta_y
        new_neg_gaze = [a / dist * new_dist for a in neg_gaze]
        new_eye = [a + b for a, b in zip(target, new_neg_gaze)]
        eye = new_eye
        glutPostRedisplay()

    previous_point = (x, y)


def main():
    global eye, target, up, fov_y, aspect, near, far, window

    window = (400, 400)
    fov_y = 40
    near = .1
    far = 50

    aspect = window[0] / window[1]
    light_position = [10., 4., 10., 1.]
    light_color = [0.8, 1.0, 0.8, 1.0]

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window[0], window[1])
    glutCreateWindow(name)

    glClearColor(0., 0., 0., 1.)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_color)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0.1)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.05)
    glEnable(GL_LIGHT0)
    glutDisplayFunc(display)

    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)

    glutMainLoop()


def display():
    global eye, target, up, fov_y, aspect, near, far

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov_y, aspect, near, far)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2],
              target[0], target[1], target[2],
              up[0], up[1], up[2])

    color = [1.0, 0., 0., 1.]
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)

    glBegin(GL_TRIANGLES)
    for v, n in zip(vertices, normals):
        glNormal3fv(list(n))
        glVertex3fv(list(v))

    glEnd()

    glutSwapBuffers()


if __name__ == '__main__':
    main()
