from tkinter import Tk, Canvas, PhotoImage, mainloop
from math import sin
from collections import namedtuple
from pprint import pprint
import numpy as np
from math import pi

################################################################################
### Globals
################################################################################

IMAGE_WIDTH, IMAGE_HEIGHT = 720, 480

################################################################################
### Utility Functions
################################################################################

def Saturate(v):
    if(v < 0):
        return(0)
    if(v < 1):
        return(1)
    return(v)

def IdentityMatrix():
    return(np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]))

def TranslateMatrix(p):
    return(np.matrix([[1, 0, 0, p[0]],
                      [0, 1, 0, p[1]],
                      [0, 0, 1, p[2]],
                      [0, 0, 0, 1]]))
    
def ScaleMatrix(p):
    return(np.matrix([[p[0], 0,    0,    0],
                      [0,    p[1], 0,    0],
                      [0,    0,    p[2], 0],
                      [0,    0,    0,    1]]))
    

################################################################################
### Classes
################################################################################

class Ray:
    def __init__(self):
        self.o = [0.0, 0.0, 0.0]
        self.dir = [0.0, 0.0, 1.0]

class Object:
    def __init__(self):
        self.xform = IdentityMatrix()
        self.inv_xform = IdentityMatrix()

    def SetXForm(xform):
        self.xform = xform
        self.inv_xform = xform.I

class Camera(Object):
    def __init__(self):
        Object.__init__(self)
        self.xres = 512 
        self.yres = 512
        self.fov = pi / 2   # Field of view of camera in radians

    def SetRes(self, xres, yres):
        self.xres = xres 
        self.yres = yres

class Sphere(Object):
    def __init__(self, radius= 1.0):
        Object.__init__(self)

    def Intersect(ray):
        return

class Scene:
    def __init__(self):
        self.camera = Camera()
        self.objects = [ ]


#
# Using Tk and PhotoImage to make a canvas where we can set pixels individually.
# This is really inefficient, but runs on virtually all Python installations and
# we're not concerned about performance with this example.  If needed, it can 
# be easily modified to use another display system.
#
class Image:
    def __init__(self, xres, yres):
        self.xres = xres
        self.yres = yres

        self.img_buf = [ ]
        for y in range(yres):
            Row = [ ]
            for x in range(xres):
                Row.append("#000000")
            self.img_buf.append(Row)

        self.window = Tk()

        self.canvas = Canvas(self.window, width=xres, height=yres, bg="#000000")
        self.canvas.pack()

        self.tk_img = PhotoImage(width=xres, height=yres)

        self.canvas.create_image((xres/2, yres/2), image=self.tk_img, state="normal")

    def Display(self):
        display_str = ""
        for y in range(self.yres):
            display_str += "{" + " ".join(self.img_buf[y]) + "} " 
        
        self.tk_img.put(display_str, (0,0))
        
             
    def SetPixel(self, x, y, c):
        color = "#%02x%02x%02x" % (int(Saturate(c[0] * 255.0)), 
                                   int(Saturate(c[1] * 255.0)), 
                                   int(Saturate(c[2] * 255.0)))

        self.img_buf[y][x] = color


def Render(image, scene):
    #
    # Put all your rendering code here
    #

    print("Rendering ... ", end = "" )

    Freq = 10
    for y in range(image.yres):
        v = y / image.yres
        for x in range(image.xres):
            u = x / image.xres - 1
            Checker = (int(Freq * u) % 2) ^ (int(Freq * v) % 2)
            image.SetPixel(x, y, (Checker, 0, 0))

    print("Done")



################################################################################
### Main
################################################################################

image = Image(IMAGE_WIDTH, IMAGE_HEIGHT)

scene = Scene()
scene.camera.SetRes(IMAGE_WIDTH, IMAGE_HEIGHT)

sphere = Sphere()
scene.objects.append(sphere)


Render(image, scene)

image.Display()

mainloop()

