import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure
import re
import sys
from matplotlib import collections as mc
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import math
from math import cos, sin
import matplotlib.animation
import pandas as pd
import copy

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
whole_data = []
color_values = []

def read_reflectivity(file_name):
    sweeps = []
    metadata = []
    with open(file_name, 'rb') as fp:
        for sweep in range(1, 10):
            # read sweep delimiter
            line = fp.readline().strip().decode('utf-8')
            header = 'SWEEP%dRFLCTVTY' % sweep
            if line != header:
                print('Error: Failed to find "%s" in "%s"' % (header, line))
                return

            # print('Sweep %d' % sweep)

            # read latitude, longitude, height
            line = fp.readline().strip().decode('utf-8')
            # print(line)
            tokens = line.split()
            if len(tokens) != 6 or tokens[0] != 'Latitude:' or tokens[2] != 'Longitude:' or tokens[4] != 'Height:':
                print('Error: Failed to find Lat, Lon, Ht in %s' % tokens)
                return
            latitude = float(tokens[1])
            longitude = float(tokens[3])
            height = float(tokens[5])
            # print('lat', latitude, 'lon', longitude, 'height', height)

            # read number of radials
            num_radials = int(fp.readline().strip().decode('utf-8'))
            # print(num_radials, 'radials')

            gate_dist = float(fp.readline().strip().decode('utf-8'))
            # print(gate_dist, 'meters to gate')

            sweep_data = {
                'latitude': latitude,
                'longitude': longitude,
                'height': height,
                'num_radials': num_radials,
                'gate_dist': gate_dist
            }

            data = []
            radial_data = []
            for radial in range(num_radials):
                #print('for radial %d out of %d' % (radial, num_radials))
                tokens = fp.readline().strip().split()
                current_radial, num_gates, gate_width = (int(t) for t in tokens[:3])
                beam_width, azimuth, elevation = [float(t) for t in tokens[3:-1]]
                start_time = int(tokens[-1])
                #print(current_radial, num_gates, gate_width, beam_width, azimuth, elevation, start_time)
                empty_line = fp.readline().strip().decode('utf-8')
                if empty_line != '':
                    raise (Exception('Error: no empty line'))

                seconds_since_epoch = fp.readline().strip().decode('utf-8')
                if seconds_since_epoch != 'seconds since epoch':
                    raise (Exception('Error: no "seconds since epoch"'))

                x = np.fromfile(fp, dtype='>f', count=num_gates)
                x[x < 0] = 0
                data.append(x)
                radial_data.append({
                    'beam_width': beam_width,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'start_time': start_time,
                })
            data = np.array(data)
            data = data.T
            sweeps.append(np.array(data))
            metadata.append({
                'sweep': sweep_data,
                'radials': radial_data
            })

        sweeps = np.array(sweeps)
        for i in range(len(sweeps)):
            print('sweep %d: [%g, %g], %g +/- %g' % (
                i, sweeps[i].min(), sweeps[i].max(), sweeps[i].mean(), sweeps[i].std()))

    return sweeps, metadata

def opengliso(sweeps):
    global eye, target, up, fov_y, aspect, near, far, window, image_width, image_height, image_depth, win_id, points
    image_width = len(sweeps)
    image_height = len(sweeps[0])
    image_depth = len(sweeps[0][0])
    eye = [(image_width - 1), (image_height - 1), 2 * image_depth]
    target = [(image_width - 1) / 2, (image_height - 1) / 2, (image_depth - 1) / 2]
    up = [0, 1, 0]

    window = (1000, 1000)
    fov_y = 40
    near = .1
    far = 1000

    aspect = window[0] / window[1]
    light_position = eye
    light_color = [100.0, 100.0, 100.0, 1.0]
    create_mesh(sweeps)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window[0], window[1])
    win_id = glutCreateWindow('weather')

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

    glutDisplayFunc(display2)
    glutMouseFunc(mouse_func)
    glutMotionFunc(motion_func)
    glutMainLoop()

def create_mesh(sweeps):
    global vertices, normals, triangles, points, image_height, image_width, image_depth, points

    threshold = 5
  #  # using rotations here: http://www.euclideanspace.com/maths/geometry/rotations/axisAngle/examples/index.htm
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
    points = []

    # TODO: Fill in vertices and normals for each triangle here
    for i in range(len(sweeps)):
        for j in range(len(sweeps[i])):
            for k in range(len(sweeps[i][j])):
                if (i + 1 < len(sweeps) and j + 1 < len(sweeps[i+1]) and k + 1 < len(sweeps[i+1][j+1])) and (sweeps[i][j][k] > threshold or sweeps[i][j][k+1] > threshold or sweeps[i][j+1][k] > threshold or sweeps[i][j+1][k+1] > threshold or sweeps[i+1][j][k] > threshold or sweeps[i+1][j+1][k] > threshold or sweeps[i+1][j+1][k+1] > threshold):
                    # col_values = []
                    if(sweeps[i][j][k] > threshold):
                        points.append(([np.cos(np.radians(k)) * j, np.sin(np.radians(k)) * j, j], [0, 1, 0]))
                    m = np.sin(np.radians(j)) * i
                    y = np.cos(np.radians(j)) * i
                    counter0 = 0
                    counter = 0
                    counter0 = check_cases_array(sweeps,i,j,k)
                    #run while to rotate and use check cases function again and check base cases
                    #top = [x[i][j][k], x[i][j + 1][k], x[i + 1][j][k], x[i + 1][j + 1][k]]
                    #bottom = [x[i][j][k+1], x[i][j + 1][k+1], x[i + 1][j][k+1], x[i + 1][j + 1][k+1]]
                    #indexes = [(i,j,k),(i,j+1,k),(i + 1,j,k), (i + 1, j + 1, k), (i,j,k + 1), (i, j + 1, k + 1), (i + 1, j, k + 1), (i + 1, j + 1, k + 1)]
                    for m in range(0,22):
                        if(counter0 > -1):
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([-1, 0, -1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([0, -1, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([0, -1, -1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1) / 2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([-1, -1, 0])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([0, -1, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([-1, 0, 1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([0, 1, -1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([0, -1, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([1, -1, 0])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([0, 1, -1])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([1, -1, 0])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array((1, 0, -1))
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([1, 0, -1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([-1, 0, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([1, 0, 1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([0, 1, -1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([0, -1, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([1, -1, 0])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([0, 1, -1])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([1, -1, 0])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([1, 0, -1])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
                                ver9 = np.array([-1, -1, 0])
                                ver9 = ver9.dot(rotations[m])
                                ver10 = np.array([0, -1, -1])
                                ver10 = ver10.dot(rotations[m])
                                ver11 = np.array([-1, 0, -1])
                                ver11 = ver11.dot(rotations[m])
                                ver9 = (ver9 + 1)/2
                                ver10 = (ver10 + 1) / 2
                                ver11 = (ver11 + 1) / 2
                                vertices.append(ver9 + np.array([m,y,k]))
                                vertices.append(ver10 + np.array([m,y,k]))
                                vertices.append(ver11 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([0, 1, 1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([1, 1, 0])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([1, 0, 1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([-1, 0, 1])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([-1, -1, 0])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([0, -1, 1])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
                                ver9 = np.array([1, 0, -1])
                                ver9 = ver9.dot(rotations[m])
                                ver10 = np.array([0, -1, -1])
                                ver10 = ver10.dot(rotations[m])
                                ver11 = np.array([1, -1, 0])
                                ver11 = ver11.dot(rotations[m])
                                ver9 = (ver9 + 1)/2
                                ver10 = (ver10 + 1) / 2
                                ver11 = (ver11 + 1) / 2
                                vertices.append(ver9 + np.array([m,y,k]))
                                vertices.append(ver10 + np.array([m,y,k]))
                                vertices.append(ver11 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([0, 1, -1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([-1, 0, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([0, -1, 1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([0, 1, -1])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([0, -1, 1])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([1, 0, -1])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
                                ver9 = np.array([1, -1, 0])
                                ver9 = ver9.dot(rotations[m])
                                ver10 = np.array([1, 0, -1])
                                ver10 = ver10.dot(rotations[m])
                                ver11 = np.array([0, -1, 1])
                                ver11 = ver11.dot(rotations[m])
                                ver9 = (ver9 + 1)/2
                                ver10 = (ver10 + 1) / 2
                                ver11 = (ver11 + 1) / 2
                                vertices.append(ver9 + np.array([m,y,k]))
                                vertices.append(ver10 + np.array([m,y,k]))
                                vertices.append(ver11 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([-1, 1, 0])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([1, 0, -1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([0, 1, -1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([-1, 1, 0])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([-1, -1, 0])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([0, -1, 1])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
                                ver9 = np.array([1, 0, -1])
                                ver9 = ver9.dot(rotations[m])
                                ver10 = np.array([0, -1, 1])
                                ver10 = ver10.dot(rotations[m])
                                ver11 = np.array([1, 0, 1])
                                ver11 = ver11.dot(rotations[m])
                                ver9 = (ver9 + 1)/2
                                ver10 = (ver10 + 1) / 2
                                ver11 = (ver11 + 1) / 2
                                vertices.append(ver9 + np.array([m,y,k]))
                                vertices.append(ver10 + np.array([m,y,k]))
                                vertices.append(ver11 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([0, 1, -1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([1, 1, 0])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([1, 0, -1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([1, -1, 0])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([-1, 0, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([-1, -1, 0])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([1, -1, 0])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([1, 0, 1])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([-1, 0, 1])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([0, 1, 1])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([-1, 1, 0])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([-1, 0, 1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([0, 1, -1])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([1, 1, 0])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([1, 0, -1])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([-1, -1, 0])
                                ver3 = ver3.dot(rotations[m])
                                ver4 = np.array([0, -1, 1])
                                ver4 = ver4.dot(rotations[m])
                                ver5 = np.array([0, 1, 1])
                                ver5 = ver5.dot(rotations[m])
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([0, 1, -1])
                                ver6 = ver6.dot(rotations[m])
                                ver7 = np.array([1, 1, 0])
                                ver7 = ver7.dot(rotations[m])
                                ver8 = np.array([1, -1, 0])
                                ver8 = ver8.dot(rotations[m])
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
                                ver9 = np.array([0, 1, -1])
                                ver9 = ver9.dot(rotations[m])
                                ver10 = np.array([0, -1, -1])
                                ver10 = ver10.dot(rotations[m])
                                ver11 = np.array([1, -1, 0])
                                ver11 = ver11.dot(rotations[m])
                                ver9 = (ver9 + 1)/2
                                ver10 = (ver10 + 1) / 2
                                ver11 = (ver11 + 1) / 2
                                vertices.append(ver9 + np.array([m,y,k]))
                                vertices.append(ver10 + np.array([m,y,k]))
                                vertices.append(ver11 + np.array([m,y,k]))
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
                                vertices.append(ver0 + np.array([m,y,k]))
                                vertices.append(ver1 + np.array([m,y,k]))
                                vertices.append(ver2 + np.array([m,y,k]))
                                ver3 = np.array([-1, 0, -1])
                                ver3 = ver3.dot(rotations[m].T)
                                ver4 = np.array([-1, 0, 1])
                                ver4 = ver4.dot(rotations[m].T)
                                ver5 = np.array([0, -1, 1])
                                ver5 = ver5.dot(rotations[m].T)
                                ver3 = (ver3 + 1)/2
                                ver4 = (ver4 + 1) / 2
                                ver5 = (ver5 + 1) / 2
                                vertices.append(ver3 + np.array([m,y,k]))
                                vertices.append(ver4 + np.array([m,y,k]))
                                vertices.append(ver5 + np.array([m,y,k]))
                                ver6 = np.array([1, 1, 0])
                                ver6 = ver6.dot(rotations[m].T)
                                ver7 = np.array([0, 1, -1])
                                ver7 = ver7.dot(rotations[m].T)
                                ver8 = np.array([-1, 0, -1])
                                ver8 = ver8.dot(rotations[m].T)
                                ver6 = (ver6 + 1)/2
                                ver7 = (ver7 + 1) / 2
                                ver8 = (ver8 + 1) / 2
                                vertices.append(ver6 + np.array([m,y,k]))
                                vertices.append(ver7 + np.array([m,y,k]))
                                vertices.append(ver8 + np.array([m,y,k]))
                                ver9 = np.array([1, 1, 0])
                                ver9 = ver9.dot(rotations[m].T)
                                ver10 = np.array([0, -1, 1])
                                ver10 = ver10.dot(rotations[m].T)
                                ver11 = np.array([1, -1, 0])
                                ver11 = ver11.dot(rotations[m].T)
                                ver9 = (ver9 + 1)/2
                                ver10 = (ver10 + 1) / 2
                                ver11 = (ver11 + 1) / 2
                                vertices.append(ver9 + np.array([m,y,k]))
                                vertices.append(ver10 + np.array([m,y,k]))
                                vertices.append(ver11 + np.array([m,y,k]))
                                break
    points = np.array(points)

    for i in range(len(vertices)):
        normals.append((0, 0, 1))

    vertices = np.array(vertices)
    normals = np.array(normals)


def invert(h):
    powers = [2**i for i in range(7, -1, -1)]
    return np.array([1-int(a) for a in list('{0:08b}'.format(h))]).dot(powers)

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

def rotated(array_2d):
    list_of_tuples = zip(*array_2d[::-1])
    return [list(elem) for elem in list_of_tuples]

def check_cases_array(x,i,j,k):
    if(i+1 < len(x) and j+1 < len(x[i + 1]) and k+1 < len(x[i + 1][j + 1])):
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
    else:
        counter = -1

def openglplot(sweeps):
    global eye, target, up, fov_y, aspect, near, far, window, image_width, image_height, image_depth, win_id, points
    image_width = len(sweeps)
    image_height = len(sweeps[0])
    image_depth = len(sweeps[0][0])
    eye = [(image_width - 1), (image_height - 1), 2 * image_depth]
    target = [(image_width - 1) / 2, (image_height - 1) / 2, (image_depth - 1) / 2]
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
    win_id = glutCreateWindow('weather')

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
    points = []
    for i in range(len(sweeps)):
        for j in range(len(sweeps[i])):
            for k in range(len(sweeps[i][j])):
                if(sweeps[i][j][k] > 15):
                    points.append(([np.cos(np.radians(k)) * j, np.sin(np.radians(k)) * j, j], [0, 0, 1]))
    points = np.array(points)

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
    # glBegin(GL_TRIANGLES)
    # for i in range(len(vertices)):
    #     glColor3fv([1, 0, 0])
    #     glNormal3fv(normals[i, :])
    #     glVertex3fv(vertices[i, :])
    # glEnd()

    glPointSize(10)
    glBegin(GL_POINTS)
    for point, c in points:
        glColor3fv(c)
        glVertex3fv(point)
    glEnd()
    glEnable(GL_LIGHTING)

    glutSwapBuffers()

def display2():
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
    #glPointSize(10)
    # glBegin(GL_POINTS)
    # for point, c in points:
    #     glColor3fv(c)
    #     glVertex3fv(point)
    # glEnd()
    glEnable(GL_LIGHTING)

    glutSwapBuffers()

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

def threedscatterforall(sweeps):
    x_values = []
    y_values = []
    z_values = []
    col_values = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for m in range(len(sweeps)):
        for i in range(len(sweeps[m])):
            for j in range(len(sweeps[m][i])):
                if(sweeps[m][i][j] > 15):
                    x_values.append(np.cos(np.radians(j))*i)
                    y_values.append(np.sin(np.radians(j))*i)
                    z_values.append(i)
                    col_values.append(sweeps[m][i][j])

    #plt.scatter(x_values,y_values,c=col_values)
    #plt.imshow(sweeps[sweep])
    ax.scatter(x_values,y_values,z_values,c=col_values)
    #plt.colorbar()
    #plt.xlabel('angle')
    #plt.ylabel('distance')
    plt.show()

def threedscatter(sweep):
    x_values = []
    y_values = []
    z_values = []
    col_values = []
    ax = plt.subplot(111, projection='3d')
    for i in range(len(sweep)):
        for j in range(len(sweep[i])):
            x_values.append(np.cos(np.radians(j))*i)
            y_values.append(np.sin(np.radians(j))*i)
            z_values.append(i)
            col_values.append(sweep[i][j])

    #plt.scatter(x_values,y_values,c=col_values)
    #plt.imshow(sweeps[sweep])
    ax.scatter(x_values,y_values,z_values,c=col_values)
    #plt.colorbar()
    #plt.xlabel('angle')
    #plt.ylabel('distance')
    plt.show()

def contour(sweep):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    lines = []
    threshold = 13
    x_values = []
    y_values = []
    col_values = []
    for x in range(len(sweep)):
        for y in range(len(sweep[x])):
            x_values.append(np.cos(np.radians(y)) * x)
            y_values.append(np.sin(np.radians(y)) * x)
            i = np.sin(np.radians(y)) * x
            j = np.cos(np.radians(y)) * x
            col_values.append(sweep[x][y])
            if (x + 1 < len(sweep) and y + 1 < len(sweep[x])):
                counter = 0
                if (sweep[x + 1][y] > threshold):
                    counter = counter + 1
                if (sweep[x + 1][y + 1] > threshold):
                    counter = counter + 2
                if (sweep[x][y + 1] > threshold):
                    counter = counter + 4
                if (sweep[x][y] > threshold):
                    counter = counter + 8
                if (counter == 1 or counter == 14):
                    lines.append([(1 + j, 0 + i), (2 + j, 1 + i)])
                elif (counter == 2 or counter == 13):
                    lines.append([(0 + j, 1 + i), (1 + j, 0 + i)])
                elif (counter == 3 or counter == 12):
                    lines.append([(-1 + j, 0 + i), (1 + j, 0 + i)])
                elif (counter == 4 or counter == 11):
                    lines.append([(0 + j, 1 + i), (1 + j, 0 + i)])
                elif (counter == 5):
                    lines.append([(-1 + j, 0 + i), (0 + j, 1 + i)])
                    lines.append([(0 + j, -1 + i), (1 + j, 0 + i)])
                elif (counter == 6 or counter == 9):
                    lines.append([(0 + j, 1 + i), (0 + j, -1 + i)])
                elif (counter == 7 or counter == 8):
                    lines.append([(-1 + j, 0 + i), (0 + j, 1 + i)])
                elif (counter == 10):
                    lines.append([(-1 + j, 0 + i), (0 + j, -1 + i)])
                    lines.append([(0 + j, 1 + i), (1 + j, 0 + i)])
    ax2 = fig.add_subplot(1,1,1)
    ax2.scatter(x_values, y_values, c=col_values)
    ax2.autoscale()


    lc = mc.LineCollection(lines, colors="red", linewidths=1)
    ax.add_collection(lc)
    ax.autoscale()

    # contours = measure.find_contours(x, 40)

    # fig, ax = plt.subplots()
    # ax.imshow(x, cmap=plt.cm.gray)

    # for n, contour in enumerate(contours):
    #    ax.plot(contour[:, 1], contour[:, 0], linewidth=1.8, color='red')

    # ax.axis('image')

    #im = ax.imshow(sweep, cmap=plt.cm.gray)

    plt.colorbar(plt.scatter(x_values,y_values,c=col_values))
    plt.show()

def twodscatter(sweep):
    x_values = []
    y_values = []
    col_values = []
    for i in range(len(sweep)):
        for j in range(len(sweep[i])):
            x_values.append(np.cos(np.radians(j))*i)
            y_values.append(np.sin(np.radians(j))*i)
            col_values.append(sweep[i][j])
    plt.clf()
    plt.scatter(x_values,y_values,c=col_values)
    #plt.imshow(sweeps[sweep])
    plt.colorbar()
    plt.xlabel('angle')
    plt.ylabel('distance')
    plt.show()

def overtime():
    def update_graph(num):
        global whole_data
        graph._offsets3d = (whole_data[num][0], whole_data[num][1], whole_data[num][2])
        graph._facecolors = (color_values[num])
        title.set_text('Weather Data, rflctvty_file={}'.format(num + 100))

    global whole_data,color_values
    x_values = []
    y_values = []
    z_values = []
    color_values = []
    whole_data = []
    for i in range(100):
        num = i + 100
        x_values = []
        y_values = []
        z_values = []
        colors = []
        print(num)
        if num != 131:
            file_name = './data/weather/%d.RFLCTVTY' % num
            sweeps, metadata = read_reflectivity(file_name)
            for m in range(len(sweeps)):
                for i in range(len(sweeps[m])):
                    for j in range(len(sweeps[m][i])):
                        if (sweeps[m][i][j] > 30):
                            x_values.append(np.cos(np.radians(j)) * i)
                            y_values.append(np.sin(np.radians(j)) * i)
                            z_values.append(i)
                            colors.append(sweeps[m][i][j])
            whole_data.append([x_values,y_values,z_values])
            color_values.append(colors)
    whole_data = np.array(whole_data)
    color_values = np.array(color_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('Weather Data')
    graph = ax.scatter(whole_data[0][0],whole_data[0][1],whole_data[0][2])
    title.set_text('Weather Data, rflctvty_file={}'.format(100))

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, 100,
                                             interval=50, blit=False)

    plt.show()

def main():
    #Part A:
    # index = 121
    # file_name = './data/weather/%d.RFLCTVTY' % index
    # sweeps, metadata = read_reflectivity(file_name)
    #twodscatter(sweeps[0])

    #Part B:
    # index = 121
    # file_name = './data/weather/%d.RFLCTVTY' % index
    # sweeps, metadata = read_reflectivity(file_name)
    #contour(sweeps[0])

    #Part C:
    # index = 121
    # file_name = './data/weather/%d.RFLCTVTY' % index
    # sweeps, metadata = read_reflectivity(file_name)
    #threedscatter(sweeps[0])

    #Part D:
    # index = 121
    # file_name = './data/weather/%d.RFLCTVTY' % index
    # sweeps, metadata = read_reflectivity(file_name)
    #threedscatterforall(sweeps)

    #Part E:
    # index = 121
    # file_name = './data/weather/%d.RFLCTVTY' % index
    # sweeps, metadata = read_reflectivity(file_name)
    # openglplot(sweeps)

    #Part F:
    # index = 121
    # file_name = './data/weather/%d.RFLCTVTY' % index
    # sweeps, metadata = read_reflectivity(file_name)
    #opengliso(sweeps)

    #Part 2:
    overtime()




if __name__ == '__main__':
    main()
