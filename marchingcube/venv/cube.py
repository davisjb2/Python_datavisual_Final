import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

verticies = (
    (1, -1, -1), #bottomrightback
    (1, 1, -1), #toprightback
    (-1, 1, -1), #topleftback
    (-1, -1, -1), #bottomleftback
    (1, -1, 1), #bottomright front
    (1, 1, 1), #topright front
    (-1, -1, 1), #bottomleft front
    (-1, 1, 1) #topleft front
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

colors = (
    (0.439216,0.858824,0.576471),  #aquamarine
    (0.372549,0.623529,0.623529),  #cadetblue
    (0.184314,0.309804,0.184314),  #darkgreen
    (0.576471,0.858824,0.439216),  #yellow
    (0.196078,0.8,0.196078),  #limegreen
    (1.00,0.43,0.78),  #pink
    (0.89,0.47,0.20),  #orange
    (0.36,0.20,0.09),  #chocolate
    (0.73,0.16,0.96),  #medpurple
    (0.87,0.58,0.98),  #lightpurple
    (0.439216, 0.858824, 0.576471),  # aquamarine
    (0.372549, 0.623529, 0.623529),  # cadetblue
    (0.184314, 0.309804, 0.184314),  # darkgreen
    (0.576471, 0.858824, 0.439216),  # yellow
    (0.196078, 0.8, 0.196078),  # limegreen
    (1.00, 0.43, 0.78),  # pink
    (0.89, 0.47, 0.20),  # orange
    (0.36, 0.20, 0.09),  # chocolate
    (0.73, 0.16, 0.96),  # medpurple
    (0.87, 0.58, 0.98),  #lightpurple
    (0.439216,0.858824,0.576471), #aquamarine
    (0.372549,0.623529,0.623529), #cadetblue
    (0.184314,0.309804,0.184314), #darkgreen
    (0.576471,0.858824,0.439216), #yellow
    (0.196078,0.8,0.196078), #limegreen
    )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )

def Cube():
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            #glColor3fv(colors[x])
            #glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    x = 0
    for edge in edges:
        for vertex in edge:
            x += 1
            glColor3fv(colors[x])
            glVertex3fv(verticies[vertex])
    glEnd()

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        #glRotatef(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)


main()