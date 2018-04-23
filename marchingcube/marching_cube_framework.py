from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import math
import numpy as np
from math import cos, sin


eye = None
target = None
up = None
fov_y = None
aspect = None
near = None
far = None
previous_point = None
window = None
button_down = None
vertices = []
normals = None
triangles = None
points = None
image_width = None
image_height = None
image_depth = None
win_id = None



def make_cubes():
    grid_size = 16
    x = np.zeros((grid_size * 3, grid_size * 3, 2))
    m, n, p = x.shape

    ii = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    jj = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    kk = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    for h in range(256):
        i = (h % 16) * 3
        j = int(h / 16) * 3
        x[ii+i, jj+j, kk] = [int(a)*255 for a in list('{0:08b}'.format(h))]

    for k in range(p):
        file_name = './data/cubes/%02d.pgm' % (k + 1)
        with open(file_name, 'w') as fp:
            fp.write('P2\n')
            fp.write('%d %d\n' % (n, m))
            fp.write('%d\n' % np.max(x[:, :, k]))
            for i in range(m):
                for j in range(n):
                    fp.write('%d ' % x[i, j, k])
                fp.write('\n')

    m, n, p = 2, 2, 2
    x = np.zeros((2, 2, 2))
    for h in range(256):
        x[ii, jj, kk] = [int(a)*255 for a in list('{0:08b}'.format(h))]
        file_dir = './data/cubes/%03d' % h
        if not os.path.isdir(file_dir):
            os.mkdir(file_dir)
        for k in range(2):
            file_name = '%s/%02d.pgm' % (file_dir, k + 1)
            with open(file_name, 'w') as fp:
                fp.write('P2\n')
                fp.write('%d %d\n' % (n, m))
                fp.write('%d\n' % np.max(x[:, :, k]))
                for i in range(m):
                    for j in range(n):
                        fp.write('%d ' % x[i, j, k])
                    fp.write('\n')


make_cubes()


def mouse_func(button, state, x, y):
    global previous_point, button_down
    # print(button_down, state, x, y)
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
    global win_id
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
        new_dist = dist * 2 ** (mouse_delta_y)
        new_neg_gaze = [a / dist * new_dist for a in neg_gaze]
        new_eye = [a + b for a, b in zip(target, new_neg_gaze)]
        eye = new_eye
        glutPostRedisplay()

    # print(eye)
    previous_point = (x, y)

def invert(h):
    powers = [2**i for i in range(7, -1, -1)]
    return np.array([1-int(a) for a in list('{0:08b}'.format(h))]).dot(powers)

def read_pgm(file_name):
    image = []
    with open(file_name, 'r') as fp:
        s = fp.readline().strip()
        assert s == 'P2'
        width, height = [int(x) for x in fp.readline().strip().split()]
        max_intensity = int(fp.readline().strip())

        max_int = float('-inf')
        for line in fp.readlines():
            row = [int(x) for x in line.strip().split()]
            assert len(row) == width
            image.append(row)
            max_int = max(max(row), max_int)

        assert len(image) == height
        assert max_int == max_intensity

    return image


def rot(a, x, y, z):
    # based on http://3dengine.org/Rotate_arb
    c = cos(a*np.pi/180)
    s = sin(a*np.pi/180)
    t = 1 - c
    return np.array([
        [t*x*x+c,    t*x*y-s*z,  t*x*z+s*y],
        [t*x*y+s*z,  t*y*y+c,    t*y*z-s*x],
        [t*x*z-s*y,  t*y*z+s*x,  t*z*z+c]
    ])


unit_cube = np.array([[2*int(b)-1 for b in list('{0:03b}'.format(a))] for a in range(8)])


def rotate(r, h):
    powers = [2**i for i in range(7, -1, -1)]
    i = np.argsort(((unit_cube.dot(r.T)+1)/2).dot(powers[-3:]))
    return np.array([int(a) for a in list('{0:08b}'.format(h))])[i].dot(powers)


def create_mesh():
    global vertices, normals, triangles, points, image_height, image_width, image_depth

    image_directory = sys.argv[1]
    num_images = int(sys.argv[2])
    threshold = float(sys.argv[3])

    x = []
    for i in range(1, num_images+1):
        img = read_pgm('%s/%02d.pgm' % (image_directory, i))
        x.append(img)
    x = np.array(x)
    x = np.transpose(x, (1, 2, 0))

    image_width, image_height, image_depth = x.shape
    print(image_width, image_height, image_depth)

    x = (x > threshold).astype(int)
    #print(x)
    points = []
    for i in range(image_width):
        for j in range(image_height):
            for k in range(image_depth):
                if x[i,j,k]:
                    print(i, j, k)
                    points.append(([i, j, k], [0, 0, 1]))
                else:
                    points.append(([i, j, k], [1, 0, 1]))

    points = np.array(points)
    #print(points)
    # using rotations here: http://www.euclideanspace.com/maths/geometry/rotations/axisAngle/examples/index.htm
    sq33 = 3 ** (1/2) / 3
    sq22 = 2 ** (1/2) / 2
    rotations = [
        rot(0, 1, 0, 0),       # identity
        rot(90, 1, 0, 0),      # 90 deg about x
        rot(180, 1, 0, 0),     # 180 deg about x
        rot(-90, 1, 0, 0),     # 270 deg about x
        rot(90, 0, 1, 0),      # 90 deg about y
        rot(180, 0, 1, 0),     # 180 deg about y
        rot(-90, 0, 1, 0),     # 270 deg about y
        rot(90, 0, 0, 1),      # 90 deg about z
        rot(180, 0, 0, 1),     # 180 deg about z
        rot(-90, 0, 0, 1),     # 270 deg about z
        rot(120, sq33, sq33, sq33),     # 120 deg about ( 1, 1, 1) corner 7
        rot(-120, sq33, sq33, sq33),    # 120 deg about (-1,-1,-1) corner 0
        rot(120, sq33, sq33, -sq33),    # 120 deg about ( 1, 1,-1) corner 6
        rot(-120, sq33, sq33, -sq33),   # 120 deg about (-1,-1, 1) corner 1
        rot(120, sq33, -sq33, sq33),    # 120 deg about ( 1,-1, 1) corner 5
        rot(-120, sq33, -sq33, sq33),   # 120 deg about (-1, 1,-1) corner 2
        rot(120, sq33, -sq33, -sq33),   # 120 deg about ( 1,-1,-1) corner 4
        rot(-120, sq33, -sq33, -sq33),  # 120 deg about (-1, 1, 1) corner 3
        rot(180, sq22, sq22, 0),     # 180 deg about ( 1, 1, 0) edge 23
        rot(180, 0, sq22, sq22),     # 180 deg about ( 0, 1, 1) edge 02
        rot(180, -sq22, sq22, 0),    # 180 deg about (-1, 1, 0) edge 01
        rot(180, 0, sq22, -sq22),    # 180 deg about ( 0, 1,-1) edge 13
        rot(180, sq22, 0, sq22),     # 180 deg about ( 1, 0, 1) edge 26
        rot(180, -sq22, 0, sq22),    # 180 deg about (-1, 0, 1) edge 04
    ]

    missed = 0

    normals = []
    vertices = []

    # TODO: Fill in vertices and normals for each triangle here
    for i in range(len(x)):
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                if (i + 1 < len(x) and j + 1 < len(x[i]) and k + 1 < len(x[i][j])):
                    counter0 = 0
                    counter = 0
                    counter0 = check_cases_array(x,i,j,k)
                    #run while to rotate and use check cases function again and check base cases
                    #top = [x[i][j][k], x[i][j + 1][k], x[i + 1][j][k], x[i + 1][j + 1][k]]
                    #bottom = [x[i][j][k+1], x[i][j + 1][k+1], x[i + 1][j][k+1], x[i + 1][j + 1][k+1]]
                    #indexes = [(i,j,k),(i,j+1,k),(i + 1,j,k), (i + 1, j + 1, k), (i,j,k + 1), (i, j + 1, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1)]
                    for m in range(0,22):
                        counter = rotate(rotations[m],counter0)
                        print("counter: ", counter)
                        icounter = invert(counter)
                        if (counter == 1):
                            ver0 = np.array([-1, -1, 0])
                            #ver0 = ver0.dot(rotations[1].T)
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, -1, -1])
                            #ver1 = ver1.dot(rotations[1].T)
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, 0, -1])
                            #ver2 = ver2.dot(rotations[1].T)
                            ver2 = ver2.dot(rotations[m])
                            print(ver0)
                            print(ver1)
                            print(ver2)
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            break
                        elif (counter == 3):
                            ver0 = np.array([-1, 0, 1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, -1, 1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, 0, -1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([-1, 0, -1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([0, -1, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([0, -1, -1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1) / 2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            break
                        elif (counter == 6):
                            ver0 = np.array([-1, 1,0])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, 1, -1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, 0, -1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([-1, -1, 0])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([0, -1, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([-1, 0, 1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            break
                        elif (counter == 208):
                            ver0 = np.array([0, 1, -1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, 1, 1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array((0, -1, 1))
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([0, 1, -1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([0, -1, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([1, -1, 0])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([0, 1, -1])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([1, -1, 0])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array((1, 0, -1))
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            break
                        elif (counter == 51):
                            ver0 = np.array([1, 0, -1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([-1, 0, -1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, 0, 1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([1, 0, -1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([-1, 0, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([1, 0, 1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            break
                        elif (counter == 54):
                            ver0 = np.array([0, 1, -1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, 1, 1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([0, -1, 1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([0, 1, -1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([0, -1, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([1, -1, 0])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([0, 1, -1])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([1, -1, 0])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([1, 0, -1])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            ver9 = np.array([-1, -1, 0])
                            ver9 = ver9.dot(rotations[m])
                            ver10 = np.array([0, -1, -1])
                            ver10 = ver10.dot(rotations[m])
                            ver11 = np.array([-1, 0, -1])
                            ver11 = ver11.dot(rotations[m])
                            ver9 = (ver9 + 1)/2
                            ver10 = (ver10 + 1) / 2
                            ver11 = (ver11 + 1) / 2
                            vertices.append(ver9 + np.array([i,j,k]))
                            vertices.append(ver10 + np.array([i,j,k]))
                            vertices.append(ver11 + np.array([i,j,k]))
                            break
                        elif (counter == 150):
                            ver0 = np.array([-1, 1, 0])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, 1, -1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, 0, -1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([0, 1, 1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([1, 1, 0])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([1, 0, 1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([-1, 0, 1])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([-1, -1, 0])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([0, -1, 1])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            ver9 = np.array([1, 0, -1])
                            ver9 = ver9.dot(rotations[m])
                            ver10 = np.array([0, -1, -1])
                            ver10 = ver10.dot(rotations[m])
                            ver11 = np.array([1, -1, 0])
                            ver11 = ver11.dot(rotations[m])
                            ver9 = (ver9 + 1)/2
                            ver10 = (ver10 + 1) / 2
                            ver11 = (ver11 + 1) / 2
                            vertices.append(ver9 + np.array([i,j,k]))
                            vertices.append(ver10 + np.array([i,j,k]))
                            vertices.append(ver11 + np.array([i,j,k]))
                            break
                        elif (counter == 23):
                            ver0 = np.array([0, 1, -1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([-1, 1, 0])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, 0, 1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([0, 1, -1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([-1, 0, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([0, -1, 1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([0, 1, -1])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([0, -1, 1])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([1, 0, -1])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            ver9 = np.array([1, -1, 0])
                            ver9 = ver9.dot(rotations[m])
                            ver10 = np.array([1, 0, -1])
                            ver10 = ver10.dot(rotations[m])
                            ver11 = np.array([0, -1, 1])
                            ver11 = ver11.dot(rotations[m])
                            ver9 = (ver9 + 1)/2
                            ver10 = (ver10 + 1) / 2
                            ver11 = (ver11 + 1) / 2
                            vertices.append(ver9 + np.array([i,j,k]))
                            vertices.append(ver10 + np.array([i,j,k]))
                            vertices.append(ver11 + np.array([i,j,k]))
                            break
                        elif (counter == 53):
                            ver0 = np.array([-1, 1, 0])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, -1, 1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([1, 0, -1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([-1, 1, 0])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([1, 0, -1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([0, 1, -1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([-1, 1, 0])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([-1, -1, 0])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([0, -1, 1])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            ver9 = np.array([1, 0, -1])
                            ver9 = ver9.dot(rotations[m])
                            ver10 = np.array([0, -1, 1])
                            ver10 = ver10.dot(rotations[m])
                            ver11 = np.array([1, 0, 1])
                            ver11 = ver11.dot(rotations[m])
                            ver9 = (ver9 + 1)/2
                            ver10 = (ver10 + 1) / 2
                            ver11 = (ver11 + 1) / 2
                            vertices.append(ver9 + np.array([i,j,k]))
                            vertices.append(ver10 + np.array([i,j,k]))
                            vertices.append(ver11 + np.array([i,j,k]))
                            break
                        elif (counter == 66):
                            ver0 = np.array([0, -1, 1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([-1, 0, 1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, -1, 0])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([0, 1, -1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([1, 1, 0])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([1, 0, -1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            break
                        elif (counter == 98):
                            ver0 = np.array([0, 1, -1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([1, 1, 0])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([1, 0, -1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([1, -1, 0])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([-1, 0, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([-1, -1, 0])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([1, -1, 0])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([1, 0, 1])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([-1, 0, 1])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            break
                        elif (counter == 104):
                            ver0 = np.array([0, -1, 1])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([1, -1, 0])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([1, 0, 1])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([0, 1, 1])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([-1, 1, 0])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([-1, 0, 1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([0, 1, -1])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([1, 1, 0])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([1, 0, -1])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            break
                        elif (counter == 90):
                            ver0 = np.array([-1, 1, 0])
                            ver0 = ver0.dot(rotations[m])
                            ver1 = np.array([0, 1, 1])
                            ver1 = ver1.dot(rotations[m])
                            ver2 = np.array([-1, -1, 0])
                            ver2 = ver2.dot(rotations[m])
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([-1, -1, 0])
                            ver3 = ver3.dot(rotations[m])
                            ver4 = np.array([0, -1, 1])
                            ver4 = ver4.dot(rotations[m])
                            ver5 = np.array([0, 1, 1])
                            ver5 = ver5.dot(rotations[m])
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([0, 1, -1])
                            ver6 = ver6.dot(rotations[m])
                            ver7 = np.array([1, 1, 0])
                            ver7 = ver7.dot(rotations[m])
                            ver8 = np.array([1, -1, 0])
                            ver8 = ver8.dot(rotations[m])
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            ver9 = np.array([0, 1, -1])
                            ver9 = ver9.dot(rotations[m])
                            ver10 = np.array([0, -1, -1])
                            ver10 = ver10.dot(rotations[m])
                            ver11 = np.array([1, -1, 0])
                            ver11 = ver11.dot(rotations[m])
                            ver9 = (ver9 + 1)/2
                            ver10 = (ver10 + 1) / 2
                            ver11 = (ver11 + 1) / 2
                            vertices.append(ver9 + np.array([i,j,k]))
                            vertices.append(ver10 + np.array([i,j,k]))
                            vertices.append(ver11 + np.array([i,j,k]))
                            break
                        elif (counter == 83):
                            ver0 = np.array([-1, 0, -1])
                            ver0 = ver0.dot(rotations[m].T)
                            ver1 = np.array([0, -1, 1])
                            ver1 = ver1.dot(rotations[m].T)
                            ver2 = np.array([1, 1, 0])
                            ver2 = ver2.dot(rotations[m].T)
                            ver0 = (ver0 + 1)/2
                            ver1 = (ver1 + 1) / 2
                            ver2 = (ver2 + 1) / 2
                            vertices.append(ver0 + np.array([i,j,k]))
                            vertices.append(ver1 + np.array([i,j,k]))
                            vertices.append(ver2 + np.array([i,j,k]))
                            ver3 = np.array([-1, 0, -1])
                            ver3 = ver3.dot(rotations[m].T)
                            ver4 = np.array([-1, 0, 1])
                            ver4 = ver4.dot(rotations[m].T)
                            ver5 = np.array([0, -1, 1])
                            ver5 = ver5.dot(rotations[m].T)
                            ver3 = (ver3 + 1)/2
                            ver4 = (ver4 + 1) / 2
                            ver5 = (ver5 + 1) / 2
                            vertices.append(ver3 + np.array([i,j,k]))
                            vertices.append(ver4 + np.array([i,j,k]))
                            vertices.append(ver5 + np.array([i,j,k]))
                            ver6 = np.array([1, 1, 0])
                            ver6 = ver6.dot(rotations[m].T)
                            ver7 = np.array([0, 1, -1])
                            ver7 = ver7.dot(rotations[m].T)
                            ver8 = np.array([-1, 0, -1])
                            ver8 = ver8.dot(rotations[m].T)
                            ver6 = (ver6 + 1)/2
                            ver7 = (ver7 + 1) / 2
                            ver8 = (ver8 + 1) / 2
                            vertices.append(ver6 + np.array([i,j,k]))
                            vertices.append(ver7 + np.array([i,j,k]))
                            vertices.append(ver8 + np.array([i,j,k]))
                            ver9 = np.array([1, 1, 0])
                            ver9 = ver9.dot(rotations[m].T)
                            ver10 = np.array([0, -1, 1])
                            ver10 = ver10.dot(rotations[m].T)
                            ver11 = np.array([1, -1, 0])
                            ver11 = ver11.dot(rotations[m].T)
                            ver9 = (ver9 + 1)/2
                            ver10 = (ver10 + 1) / 2
                            ver11 = (ver11 + 1) / 2
                            vertices.append(ver9 + np.array([i,j,k]))
                            vertices.append(ver10 + np.array([i,j,k]))
                            vertices.append(ver11 + np.array([i,j,k]))
                            break
                        else:
                            print("out of bounds")

    for i in range(len(vertices)):
        normals.append((0, 0, 1))

    vertices = np.array(vertices)
    normals = np.array(normals)

def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]

def check_cases_array(x,i,j,k):
    counter = 0
    if (x[i][j][k]):
        counter += 1
    if (x[i][j + 1][k]):
        counter += 4
    if (x[i + 1][j][k]):
        counter += 16
    if (x[i + 1][j + 1][k]):
        counter += 64
    if (x[i][j][k + 1]):
        counter += 2
    if (x[i][j + 1][k + 1]):
        counter += 8
    if (x[i + 1][j][k + 1]):
        counter += 32
    if (x[i + 1][j + 1][k + 1]):
        counter += 128
    return counter

def main():
    global eye, target, up, fov_y, aspect, near, far, window, image_width, image_height, image_depth, win_id
    create_mesh()

    eye = [(image_width-1), (image_height-1), 2*image_depth]
    target = [(image_width-1)/2, (image_height-1)/2, (image_depth-1)/2]
    up = [0, 1, 0]

    window = (1000, 1000)
    fov_y = 40
    near = .1
    far = 1000

    aspect = window[0] / window[1]
    light_position = eye
    light_color = [100.0, 100.0, 100.0, 1.0]

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window[0], window[1])
    win_id = glutCreateWindow('cubes')

    glClearColor(0., 0., 0., 1.)
    glShadeModel(GL_SMOOTH)
    # glEnable(GL_CULL_FACE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_color)
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 0)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0)

    glEnable(GL_PROGRAM_POINT_SIZE)

    glEnable(GL_LIGHT0)

    # callbacks
    glutDisplayFunc(display)
    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)

    glutMainLoop()


def display():
    global eye, target, up, fov_y, aspect, near, far, vertices, points, normals

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov_y, aspect, near, far)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(eye[0], eye[1], eye[2],
              target[0], target[1], target[2],
              up[0], up[1], up[2])

    glLightfv(GL_LIGHT0, GL_POSITION, eye)

    color = [1.0, 1.0, 0.0, 1.]
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)
    # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, color)
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0)

    glDisable(GL_LIGHTING)
    glBegin(GL_TRIANGLES)
    for i in range(len(vertices)):
        glColor3fv([1, 0, 0])
        glNormal3fv(normals[i, :])
        glVertex3fv(vertices[i, :])
    glEnd()

#    glPointSize(10)
#    glBegin(GL_POINTS)
#    for point, c in points:
#        glColor3fv(c)
#        glVertex3fv(point)
#    glEnd()
#    glEnable(GL_LIGHTING)

    glutSwapBuffers()



if __name__ == '__main__':
    main()