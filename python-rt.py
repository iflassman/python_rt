import numpy as np
import sys
from time import time
from tkinter import Tk, Canvas, PhotoImage, mainloop
from math import sin
from pprint import pprint
from numpy import dot as dot
from numpy import multiply as mul
from numpy import negative as neg
from numpy import matmul as matmul
from numpy import add as add
from numpy import subtract as sub
from numpy import absolute as absolute
from numpy import linalg as LA
from math import pi, cos, sin, tan, inf, sqrt, acos, atan2
from copy import deepcopy

################################################################################
### Globals
################################################################################

EPSILON = sys.float_info.epsilon
RAY_TRACE_EPSILON = .001
IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 768
MAX_RAY_RECURSION_DEPTH = 5
DEFAULT_MATERIAL = None
DEG_TO_RAD = pi / 180

################################################################################
### Utility Functions
################################################################################

def Radians(angle):
    return(DEG_TO_RAD * angle)

def Normalize(v):
    mag = np.linalg.norm(v)
    if(mag == 0.0):
        return([inf] * len(v)) 
    return(np.divide(v, mag))

def Saturate(v):
    if(v < 0):
        return(0)
    if(v > 1):
        return(1)
    return(v)

def GreaterThan3(v, a):
    return(v[0] >= a and v[1] > a and v[2] > a)

def VecMul(m, v):
    return([m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]])

def PointMul(m, p):
    return([m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3],
            m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3],
            m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3]])

################################################################################
### Transform Matrices
################################################################################

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

def RotXMatrix(angle):
    sin_angle = sin(angle)
    cos_angle = cos(angle)

    return([[1,    0,          0,         0],
            [0,    cos_angle, -sin_angle, 0],
            [0,    sin_angle,  cos_angle, 0],
            [0,    0,          0,         1]])

def RotYMatrix(angle):
    sin_angle = sin(angle)
    cos_angle = cos(angle)

    return([[cos_angle,    0,  -sin_angle, 0],
            [0,            1,  0,          0],
            [sin_angle,    0,  cos_angle,  0],
            [0,            0,  0,          1]])

def RotZMatrix(angle):
    sin_angle = sin(angle)
    cos_angle = cos(angle)

    return( [cos_angle, -sin_angle, 0, 0],
            [sin_angle,  cos_angle, 0, 0],
            [0,          0,         1, 0],
            [0,          0,         0, 1])

def ComboXForm(**kwargs):

    ret = IdentityMatrix()

    if('scale' in kwargs.keys()):
        ret = matmul(ScaleMatrix(kwargs['scale']), ret)

    if('z_angle' in kwargs.keys()):
        ret = matmul(RotZMatrix(kwargs['z_angle']), ret)
    
    if('y_angle' in kwargs.keys()):
        ret = matmul(RotYMatrix(kwargs['y_angle']), ret)

    if('x_angle' in kwargs.keys()):
        ret = matmul(RotXMatrix(kwargs['x_angle']), ret)

    if('translate' in kwargs.keys()):
        ret = matmul(TranslateMatrix(kwargs['translate']), ret)

    return(ret)


################################################################################
### Render 
################################################################################

def Render(image, scene):
    #
    # Put all your rendering code here
    #


    PercentDone = 0.0

    
    int_stack = [ ] 
    for x in range(MAX_RAY_RECURSION_DEPTH):
        new_int = Intersection(scene)
        if(len(int_stack) > 0):
            int_stack[-1].next = new_int
            new_int.prev = int_stack[-1]
            new_int.depth = new_int.prev.depth + 1
        int_stack.append(new_int)

    i = int_stack[0]
    start_time = time()
    for y in range(image.yres):
        PercentDone = 100.0 * y / image.yres
        elapsed_time = time() - start_time
    
        print("\rRendering ... %4.1f%% (%ds)" % (PercentDone, int(elapsed_time)), 
              end = "", flush = True )
        for x in range(image.xres):
            scene.camera.GenPrimaryRay(i.ray, x, y)

            if(scene.Trace(i, True)):
                image.SetPixel(x, y, i.color)
            else:
                image.SetPixel(x, y, (0, 0, 0))

    elapsed_time = time() - start_time
    print("\rRendering ... Done (%ds)          " % int(elapsed_time), flush = True)

################################################################################
### Ray and Intersection Classes
################################################################################

class Ray:
    def __init__(self, 
                 o = [0.0, 0.0, 0.0], 
                 dir = [0.0, 0.0, 1.0]):
        self.o = o
        self.dir = dir

    def Set(self, o, dir, add_eps = False):
        self.o, self.dir = [ *o ], [ *dir]
        if(add_eps):
            self.AddEpsilon()

    def AddEpsilon(self):
        self.o = add(self.o, mul(RAY_TRACE_EPSILON, self.dir))

class Intersection:
    def __init__(self, scene):
        self.depth = 0
        self.scene = scene          # Keep reference to scene
        self.ray = Ray()            # Ray used for intersection
        self.dist = inf             # distance along ray to point of intersection
        self.p = [0.0, 0.0, 0.0]    # point of intersection
        self.n = [0.0, 0.0, 0.0]    # normal at intersection
        self.uv = [0.0, 0.0]        # uv coordinates at point of intersection
        self.object = None          # Intersection object

        self.color = [1, 1, 1]      # Reflected at interesection
        self.opacity = [1, 1, 1]    # Colored opacity at intersection
        self.prev = None            # Previous intersection in recurse ray trace
        self.next = None            # Next intersection in recurse ray trace

    def CalcAll(self):
        if(self.object != None):
            self.object.CalcAllIntersection(self)

    def Save(self):
        self.save_ray = deepcopy(self.ray)
        self.save_dist = self.dist
        self.save_p = self.p
        self.save_n = self.n
        self.save_uv = [ *self.uv ]
        self.save_object = self.object

    def Restore(self):
        self.ray = self.save_ray
        self.dist = self.save_dist
        self.p = self.save_p
        self.n = self.save_n
        self.uv = self.save_uv
        self.object = self.save_object
       
################################################################################
### Textures
################################################################################

class Texture2D:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def Sample(self, uv):
        return([0.0] * 4)

class ConstTex2D(Texture2D):
    def __init__(self, color = [1, 1, 1]):
        Texture2D.__init__(self)
        self.color = color

    def Sample(self, uv):
        return(self.color)

class CheckerTex2D(Texture2D):

    def __init__(self, **kwargs):
        self.ufreq = 10
        self.vfreq = 10
        self.color0 = [0] * 3
        self.color1 = [0] * 3

        Texture2D.__init__(self, **kwargs)

    def Sample(self, uv):
        utile = int(self.ufreq * uv[0])
        vtile = int(self.vfreq * uv[1])

        if((utile) % 2 ^ (vtile % 2)):
            return(self.color0)

        return(self.color1)

################################################################################
### Materials
################################################################################

class Material:
    def __init__(self):
        return

    def Shade(self, i):
        self.color = [1, 1, 1]
        self.opacity = [1, 1, 1]

class SimpleMaterial(Material):
    def __init__(self, texture = ConstTex2D()):
        Material.__init__(self)
        self.ks =  [.7] * 3
        self.kr =  [0.0] * 3
        self.spec_exp = 15.0
        self.texture = texture

    def Shade(self, i):
        i.color = self.texture.Sample(i.uv)

        n_norm = Normalize(i.n)
        e_norm = Normalize(i.ray.dir)
        e_dot_n = dot(e_norm, n_norm)
        r = Normalize(add(e_norm, mul(-2.0 * e_dot_n, n_norm)))

        diffuse = (0.0, 0.0, 0.0)
        specular = (0.0, 0.0, 0.0)
        for light in i.scene.lights:
            light_info = light.GetLightSampleInfo(i)
            light_dist = LA.norm(light_info.dir)
            l_norm = mul(light_info.dir, 1.0 / light_dist)

            # Trace shadows
            if(i.next != None):
                i.next.ray.Set(i.p, l_norm, True)
                if(i.scene.Trace(i.next, False, True)):
                    if(i.next.dist < light_dist):
                        continue;


                l_dot_n = dot(l_norm, n_norm)
                if(l_dot_n > 0.0):
                    diffuse = add(diffuse, mul(l_dot_n, light_info.emission))

                    spec = l_dot_n * pow(dot(r, l_norm), self.spec_exp)
                    specular = add(specular, mul(spec, light_info.emission))
                
        i.color = add(mul(i.color, diffuse), mul(self.ks, specular))
        i.opacity = [1, 1, 1]

        # Trace reflection
        if(i.next != None):
            if(GreaterThan3(self.kr, .01)):
                i.next.ray.Set(i.p, r, True)
                if(i.scene.Trace(i.next, True)):
                    i.color = add(i.color, mul(i.next.color, self.kr))


DEFAULT_MATERIAL = SimpleMaterial()

################################################################################
### Objects
################################################################################

class Object:
    def __init__(self):
        self.xform = IdentityMatrix()
        self.inv_xform = IdentityMatrix()
        self.material = DEFAULT_MATERIAL

    def SetXForm(self, xform):
        self.xform = xform
        self.inv_xform = np.linalg.inv(xform)

    def Origin(self):
        return([self.xform[0][3], self.xform[1][3], self.xform[2][3]])

class Sphere(Object):
    def __init__(self):
        Object.__init__(self)

    def Intersect(self, i):

        # Transform ray into object space
        o = PointMul(self.inv_xform, i.ray.o)
        dir = VecMul(self.inv_xform, i.ray.dir)
        dir_n = Normalize(dir)


        # Assume sphere is centered at origin with radius 1.0 in object space
        b = 2.0 * dot(dir_n, o)
        c = dot(o, o) - 1

        # Use quadratic formula to solve for t
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

        t_dir = mul(t, dir_n) # scaled dir vec in global space
        t_dir_glob = VecMul(self.xform, t_dir) # in global space
        i.dist = LA.norm(t_dir_glob)
        i.p = add(i.ray.o, t_dir_glob)
        p_obj = add(o, t_dir)
        i.n = VecMul(self.xform, p_obj)
        i.uv[0] = (atan2(p_obj[2], p_obj[0]) + pi)  / (2.0 * pi) # longitude angle 
        i.uv[1] = acos(p_obj[1]) /  pi                         # latitude angle

        return True

    def CalcAllIntersection(self, i):
        return

# A rectange in the XZ plane, with ranges [-1, 1] in X and Z in object space
class Rectangle(Object):
    def __init__(self):
        Object.__init__(self)

    def Intersect(self, i):

        # Transform ray into object space
        o = PointMul(self.inv_xform, i.ray.o)
        dir = Normalize(VecMul(self.inv_xform, i.ray.dir))


        # Check if is ray is pointed at the plane
        if(o[1] > 0 and dir[1] > 0 or
           o[1] < 0 and dir[1] < 0):
            return(False)

        # No intersection if ray is parallel to XZ plane
        if abs(dir[1]) < EPSILON:
            return(False)

        ratio = -o[1] / dir[1]
        p = [ ratio * dir[0] + o[0],
               0,
              ratio * dir[2] + o[2]]

        if(p[0] < -1 or p[0] > 1):
            return(False)

        if(p[2] < -1 or p[2] > 1):
            return(False)

        p[1] = 0
        i.p = PointMul(self.xform, p)
        i.dist = LA.norm(sub(i.ray.o, i.p))
        i.n = [ self.xform[0][1], self.xform[1][1], self.xform[2][1] ]
        i.uv = [(p[0] + 1) / 2, (p[2] + 1) / 2]
        return(True)

    def CalcAllIntersection(self, i):
        return


################################################################################
### Lights
################################################################################

class LightInfo:
    def __init__(self, emission, p, dir):
        self.emission = emission
        self.p = p
        self.dir = dir

class Light(Object):
    def __init__(self, color = [1, 1, 1]):
        Object.__init__(self)
        self.color = color

class PointLight(Light):
    def __init__(self, color = [1, 1, 1]):
        Light.__init__(self, color)

    def GetLightSampleInfo(self, i):
        return(LightInfo(self.color,
                         self.Origin(),
                         sub(self.Origin(), i.p)))

################################################################################
### Camera
################################################################################

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

        # Transform ray dir into global space
        ray.Set(self.Origin(), VecMul(self.xform, dir))

    # Set Field of View in degrees
    def SetFov(self, angle): 
        self.fov = angle * pi / 180.0
       

################################################################################
### Scene
################################################################################

class Scene:
    def __init__(self):
        self.camera = Camera()
        self.objects = [ ]
        self.lights = [ ]

    def Trace(self, i, shade = False, anyhit = False, depth = 0, min_dist = inf):
        if(i == None):
            return(False)

        i.object = None
        for object in self.objects:
            if(object.Intersect(i)):
                if(anyhit):
                    return(True)

                if(i.dist < min_dist):
                    min_dist = i.dist
                    i.object = object
                    i.Save()

        if(i.object != None):
            i.Restore()
            i.CalcAll()
            i.object.material.Shade(i)
            return(True)

        return(False)

################################################################################
### Image
################################################################################

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

    def Write(self, filename):
        self.tk_img.write(filename)

    def Display(self):
        display_str = ""
        for y in reversed(range(self.yres)):
            display_str += "{" + " ".join(self.img_buf[y]) + "} " 
        
        self.tk_img.put(display_str, (0,0))
        
             
    def SetPixel(self, x, y, c):
        color = "#%02x%02x%02x" % (int(Saturate(c[0]) * 255.0), 
                                   int(Saturate(c[1]) * 255.0), 
                                   int(Saturate(c[2]) * 255.0))

        self.img_buf[int(y)][int(x)] = color



################################################################################
### Build scene
################################################################################

def BuildScene():
    scene = Scene()

    # Setup camera
    scene.camera.SetRes(IMAGE_WIDTH, IMAGE_HEIGHT)
    scene.camera.SetXForm(ComboXForm(translate = [0, 3, -5],
                                     x_angle = Radians(30)))
    scene.camera.SetFov(55)



    # Create some textures
    gray_white_tex = CheckerTex2D(color0 = [.2, .2, .2],
                                  color1 = [.8, .8, .8],
                                  ufreq = 10,
                                  vfreq = 10)

    red_checker_tex = CheckerTex2D(color0 = [0.0, 0, 0],
                                   color1 = [1.0, 0, 0],
                                   ufreq = 1,
                                   vfreq = 10)

    blue_checker_tex = CheckerTex2D(color0 = [0.0, 0, 0],
                                    color1 = [0.0, 0, 1],
                                    ufreq = 10,
                                    vfreq = 1)


    # Sphere 1
    material = SimpleMaterial(texture = blue_checker_tex)
    material.kr = [.4] * 3
    sphere = Sphere()
    sphere.material = material
    sphere.SetXForm(ComboXForm(translate = [-1.0, 0, 0.0], 
                               scale = [1] * 3, 
                               z_angle = 0))
    scene.objects.append(sphere)


    # Sphere 2
    material = SimpleMaterial(texture = red_checker_tex)
    material.color1 = [.2, .2, 1]
    material.kr = [.4] * 3
    sphere = Sphere()
    sphere.material = material
    sphere.SetXForm(ComboXForm(translate = [.7, 0, -.7], 
                               scale = [.7] * 3))
    scene.objects.append(sphere)

    # Ground plane
    material = SimpleMaterial(texture = gray_white_tex)
    material.kr = [.5] * 3
    rect = Rectangle()
    rect.material = material
    rect.SetXForm(ComboXForm(translate = [0.0, -1, 0],
                            scale = [6, 1, 6]))
    scene.objects.append(rect)

    # Light 1
    light = PointLight()
    light.SetXForm(ComboXForm(translate = [5.0, 10.0, -10]))
    light.color = [1.0] * 3
    scene.lights.append(light)

    # Light 2
    light = PointLight()
    light.SetXForm(ComboXForm(translate = [-5.0, 10.0, 1]))
    light.color = [0.2] * 3
    scene.lights.append(light)

    return(scene)

################################################################################
### Main
################################################################################

image = Image(IMAGE_WIDTH, IMAGE_HEIGHT)
scene = BuildScene()


#
# Render and display 
#
Render(image, scene)
image.Display()
image.Write('render.png')
mainloop()

