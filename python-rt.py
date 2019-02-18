import numpy as np
import sys
from tkinter import Tk, Canvas, PhotoImage, mainloop
from math import sin
from pprint import pprint
from numpy import dot as dot
from numpy import multiply as multiply
from numpy import matmul as matmul
from numpy import add as add
from numpy import subtract as subtract
from numpy import absolute as absolute
from numpy import linalg as LA
from math import pi, tan, inf, sqrt

################################################################################
### Globals
################################################################################

EPSILON = sys.float_info.epsilon
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128

################################################################################
### Utility Functions
################################################################################

def Normalize(v):
    mag = np.linalg.norm(v)
    if(mag == 0.0):
        return([inf] * len(v)) 
    return(np.divide(v, mag))

def Saturate(v):
    if(v < 0):
        return(0)
    if(v < 1):
        return(1)
    return(v)

def IdentityMatrix():
    return([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

def TranslateMatrix(p):
    return([[1, 0, 0, p[0]],
            [0, 1, 0, p[1]],
            [0, 0, 1, p[2]],
            [0, 0, 0, 1]])
    
def ScaleMatrix(p):
    return([[p[0], 0,    0,    0],
            [0,    p[1], 0,    0],
            [0,    0,    p[2], 0],
            [0,    0,    0,    1]])


def VecMul(m, v):
    return([m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]])

def PointMul(m, p):
    return([m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3]])

def Render(image, scene):
    #
    # Put all your rendering code here
    #

    print("Rendering ... ", end = "" )

    i = Intersection()

    for y in range(image.yres):
        for x in range(image.xres):
            scene.camera.GenPrimaryRay(i.ray, x, y)

            if(scene.Trace(i)):
                image.SetPixel(x, y, (1, 0, 0))
            else:
                image.SetPixel(x, y, (0, 0, 0))

    print("Done")

################################################################################
### Classes
################################################################################

class Ray:
    def __init__(self, 
                 o = [0.0, 0.0, 0.0], 
                 dir = [0.0, 0.0, 1.0]):
        self.o = o
        self.dir = dir

class Intersection:
    def __init__(self):
        self.ray = Ray()            # Ray used for intersection
        self.dist = inf             # distance along ray to point of intersection
        self.p = [0.0, 0.0, 0.0]    # point of intersection
        self.n = [0.0, 0.0, 0.0]    # normal at intersection
        self.uv = [0.0, 0.0]        # uv coordinates at point of intersection
        self.object = None          # Intersection object

    def CalcAll(self):
        if(self.object != None):
            self.object.CallAllIntersection(self)
       
class Object:
    def __init__(self):
        self.xform = IdentityMatrix()
        self.inv_xform = IdentityMatrix()

    def SetXForm(self, xform):
        self.xform = xform
        self.inv_xform = np.linalg.inv(xform)

    def Origin(self):
        return([self.xform[0][3], self.xform[1][3], self.xform[2][3]])

class Camera(Object):
    def __init__(self):
        Object.__init__(self)
        self.xres = 512 
        self.yres = 512
        self.fov = pi / 2   # Field of view of camera in radians

    def SetRes(self, xres, yres):
        self.xres = xres 
        self.yres = yres

    # Returns a ray through the center of the pixel at (x, y)
    def GenPrimaryRay(self, ray, x, y):
        half_xres = self.xres / 2
        half_yres = self.yres / 2

        focal_dist = 1.0 / tan(self.fov / 2)

        # Ray direction pointing toward center of the pixel in the image plane
        dir = [(x + 0.5 - half_xres) / half_xres,
               (y + 0.5 - half_yres) / half_xres, # assuming square pixel aspect
               focal_dist]

        ray.o = self.Origin()

        # Transform ray dir into global space
        ray.dir = VecMul(self.xform, dir)
       
class Sphere(Object):
    def __init__(self):
        Object.__init__(self)

    def Intersect(self, i):

        # Transform ray into object space
        o = PointMul(self.inv_xform, i.ray.o)
        dir = Normalize(VecMul(self.inv_xform, i.ray.dir))

        # Assume sphere is centered at origin with radius 1.0 in object space
        b = 2.0 * dot(dir, o)
        c = dot(o, o) - 1

        # Use quadric formula to solve for t
        delta = b * b - 4 * c

        if(delta < -EPSILON): # no intersection if k is negative
            return(False)

        t = 0
        if(delta < EPSILON): # intersects only once (on tangent) 
            t = -b / 2.0
        else:
            sqrt_delta = sqrt(delta)
            t0 = (-b - sqrt_delta) / 2.0
            t1 = (-b + sqrt_delta) / 2.0
            if(t0 > EPSILON and t1 > EPSILON): 
                t = min(t0, t1) # first intersection along ray
            elif(t0 > EPSILON): 
                t = t0 # t1 is behind ray
            elif(t1 > EPSILON):
                t = t1 # t0 is behind ray
            else:
                return(False) # ray intersection are both behind ray

        t_dir = multiply(t, dir) # scaled dir vec in global space
        t_dir_glob = VecMul(self.xform, t_dir) # in global space
        i.dist = LA.norm(t_dir)
        i.p = i.ray.dir + t_dir_glob
        i.n = VecMul(self.xform, add(o, t_dir))

        return True

    def CalcAllIntersection(self, i):
        return

class Scene:
    def __init__(self):
        self.camera = Camera()
        self.objects = [ ]

    def Trace(self, i):
        i.object = None
        min_dist = inf
        for object in self.objects:
            if(object.Intersect(i)):
                if(i.dist < min_dist):
                    min_dist = i.dist
                    i.object = object

        return(i.object != None)

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




################################################################################
### Main
################################################################################

image = Image(IMAGE_WIDTH, IMAGE_HEIGHT)

scene = Scene()
scene.camera.SetRes(IMAGE_WIDTH, IMAGE_HEIGHT)
scene.camera.SetXForm(TranslateMatrix([0, 0, -3]))

sphere = Sphere()
scale = ScaleMatrix([1.0, 1.0, 1.0])
translate = TranslateMatrix([-1.0, 0.0, 0.0])
xform = matmul(translate, scale)
sphere.SetXForm(xform)
scene.objects.append(sphere)

sphere = Sphere()
scale = ScaleMatrix([1.0, 2.0, 1.0])
translate = TranslateMatrix([1.0, 0.0, 0.0])
xform = matmul(translate, scale)
sphere.SetXForm(xform)
scene.objects.append(sphere)

Render(image, scene)

image.Display()

mainloop()

scene.objects.append(sphere)

sphere = Sphere()
scale = ScaleMatrix([1.0, 2.0, 2.0])
translate = TranslateMatrix([1.0, 0.0, 2.0])
xform = matmul(translate, scale)

sphere.SetXForm(xform)
scene.objects.append(sphere)

Render(image, scene)

image.Display()

mainloop()

