from tkinter import Tk, Canvas, PhotoImage, mainloop
from math import sin

################################################################################
### Globals
################################################################################

IMAGE_WIDTH, IMAGE_HEIGHT = 512, 512

################################################################################
### Functions
################################################################################

    

def Saturate(v):
    if(v < 0):
        return(0)
    if(v < 1):
        return(1)
    return(v)


def SetPixel(image, x, y, c):
    color = "#%02x%02x%02x" % (int(Saturate(c[0] * 255.0)), 
                               int(Saturate(c[1] * 255.0)), 
                               int(Saturate(c[2] * 255.0)))

    
    image[1][y][x] = color
    #image[0].put(color, (x,y))

def CreateImage(xres, yres):
    window = Tk()

    canvas = Canvas(window, width=xres, height=yres, bg="#000000")
    canvas.pack()

    tk_img = PhotoImage(width=xres, height=yres)

    canvas.create_image((xres/2, yres/2), image=tk_img, state="normal")

    img_buf = []
    for y in range(yres):
        Row = [ ]
        for x in range(xres):
            Row.append("#000000")
        img_buf.append(Row)
             

    return([tk_img, img_buf])

def DisplayImage(image):
    display_str = ""
    for y in range(IMAGE_HEIGHT):
        display_str += "{" + " ".join(image[1][y]) + "} " 
    
    image[0].put(display_str, (0,0))
    

def Render(image):
    #
    # Put all your rendering code here
    #
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            if(x == 0 or  x == IMAGE_WIDTH - 1 or
               y == 0 or  y == IMAGE_HEIGHT - 1 or
               x == y or
               x == IMAGE_HEIGHT - y):
                    SetPixel(image, x, y, (1, 0, 0))




################################################################################
### Main
################################################################################

Image = CreateImage(IMAGE_WIDTH, IMAGE_HEIGHT)

Render(Image)
DisplayImage(Image)

mainloop()

