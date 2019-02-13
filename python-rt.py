from tkinter import Tk, Canvas, PhotoImage, mainloop
from math import sin
from collections import namedtuple
from pprint import pprint

################################################################################
### Globals
################################################################################

IMAGE_WIDTH, IMAGE_HEIGHT = 720, 480

################################################################################
### Classes
################################################################################

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
### Functions
################################################################################

    

def Saturate(v):
    if(v < 0):
        return(0)
    if(v < 1):
        return(1)
    return(v)




def Render(image):
    #
    # Put all your rendering code here
    #

    image_xres = image.xres
    image_yres = image.yres 

    print("Rendering ... ", end = "" )
    Freq = 10
    for y in range(image_yres):
        v = y / image_yres
        for x in range(image_xres):
            u = x / image_xres - 1
            Checker = (int(Freq * u) % 2) ^ (int(Freq * v) % 2)
            image.SetPixel(x, y, (Checker, 0, 0))
    print("Done")



################################################################################
### Main
################################################################################

image = Image(IMAGE_WIDTH, IMAGE_HEIGHT)

Render(image)

image.Display()

mainloop()

