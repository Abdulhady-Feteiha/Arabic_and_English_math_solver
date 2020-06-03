from tkinter import *
from PIL import Image, ImageDraw
import time

width = 800
height = 500
center = height//2
white = (255, 255, 255)
green = (0,128,0)

root = Tk()
def submitcallback():
    save_as_png(cv,'hi')
def save_as_png(canvas,fileName):
    # save postscipt image 
    canvas.postscript(file = fileName + '.eps') 
    # use PIL to convert to PNG 
    img = Image.open(fileName + '.eps') 
    img.save(fileName + '.png', 'png')
# do the Tkinter canvas drawings (visible)
def paint( event ):
   python_green = "#476042"
   x1, y1 = ( event.x - 1 ), ( event.y - 1 )
   x2, y2 = ( event.x + 1 ), ( event.y + 1 )
   cv.create_oval( x1, y1, x2, y2, fill = python_green,width=12 )
def delete_paint(event):
    cv.delete(ALL)


# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack(expand = YES, fill = BOTH)

# PIL create an empty image and draw object to draw on
# memory only, not visible
image1 = Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)


cv.bind( "<B1-Motion>", paint )
cv.bind( "<Button-3>",delete_paint)



button = Button(
    root,
    text="submit",command = submitcallback)
button.pack()

root.mainloop()
