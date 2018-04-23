from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import math
# import numpy as np

name = 'ball_glut'


v0 = (+1, -1, -1) #bottomrightback
v1 = (+1, +1, -1) #toprightback
v2 = (-1, +1, -1) #topleftback
v3 = (-1, -1, -1) #bottomleftback
v4 = (+1, -1, +1) #bottomrightfront
v5 = (+1, +1, +1) #toprightfront
v6 = (-1, +1, +1) #topleftfront
v7 = (-1, -1, +1) #bottomleftfront

e0 = (+2, -2, -.5) #bottomrightback
e02 = (+.5, -2, -2) #bottomrightback
e1 = (+2, +.5, -2) #toprightback
e12 = (+.5, +2, -2) #toprightback
e13 = (+2, +2, -.5) #toprightback
e2 = (-2, +.5, -2) #topleftback
e22 = (-2, +2, -1) #topleftback
e3 = (-2, -2, -.5) #bottomleftback
e4 = (+.5, -2, +2) #bottomrightfront
e42 = (+2, -.5, +2) #bottomrightfront
e5 = (+.5, +2, +2) #toprightfront
e52 = (+2, +.5, +2) #toprightfront
e6 = (-2, +.5, +2) #bottomleftfront
e7 = (-2, -.5, +2) #topleftfront
e72 = (-2, -.02, +2) #topleftfront
e53 = (+.5, +2, +2) #topleftfront

verticies = (
    (2, -2, -2), #bottomrightback
    (2, 2, -2), #toprightback
    (-2, 2, -2), #topleftback
    (-2, -2, -2), #bottomleftback
    (2, -2, 2), #bottomright front
    (2, 2, 2), #topright front
    (-2, -2, 2), #bottomleft front
    (-2, 2, 2) #topleft front
    )

edges = (
    (0,1), #right
    (0,3), #bottom
    (0,4), #cbottomright
    (2,1), #top
    (2,3), #left
    (2,7), #ctopleft
    (6,3), #cbottomleft
    (6,4), #bottom
    (6,7), #left
    (5,1), #ctopright
    (5,4), #right
    (5,7)  #top
    )

right = (1, 0, 0)
left = (-1, 0, 0)
top = (0, 1, 0)
bottom = (0, -1, 0)
front = (0, 0, 1)
back = (0, 0, -1)

vertices = [
    #e4, e7, e3, #case1
    #e0, e7, e3, #case2
    #e0, e42, e7, #case2
    #e4, e7, e3, #case3
    #e1, e5, e42, #case3
    #e2, e42, e1, #case4
    #e2, e3, e42, #case4
    #e4, e42, e3, #case4
    #e1, e2, e6, #case5
    #e1, e6, e52, #case5
    #e2, e42, e1, #case6
    #e2, e3, e42, #case6
    #e4, e42, e3, #case6
    #e53, e22, e72, #case6
    #e4, e7, e3, #case7
    #e13, e5, e42, #case7
    #e12, e22, e2, #case7
    #e1, e02, e0, #case7
    #e12, e22, e72, #case8
    #e12, e72, e4, #case8
    #e12, e4, e1, #case8
    #e0, e1, e4, #case8
    #e22,e4,e1, #case9
    #e22,e1,e12, #case9
    #e22,e3,e4, #case9
    #e1,e4,e42, #case9
    #e4, e7, e3, #case10
    #e12, e1, e13, #case10
    #e12, e1, e13, #case11
    #e0, e7, e3, #case11
    #e0, e42, e7, #case11
    #e12, e1, e13, #case12
    #e53, e22, e72, #case12
    #e0, e42, e4, #case12
    #e4, e22, e3, #case13
    #e4, e5, e22, #case13
    #e0, e12, e02, #case13
    #e0, e13, e12, #case13
    #e2,e4,e13, #case14
    #e2,e7,e4, #case14
    #e13,e12,e2, #case14
    #e13,e4,e0, #case14

]

normals = [
    front, front, front,
    top, top, top,
    front, front, front,
    front, front, front,
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
        print(list(v))
        glNormal3fv(list(n))
        glVertex3fv(list(v))

    glEnd()
    glBegin(GL_LINES)
    x = 0
    for edge in edges:
        for vertex in edge:
            x += 1
            glVertex3fv(verticies[vertex])
    glEnd()

    glutSwapBuffers()


if __name__ == '__main__':
    main()