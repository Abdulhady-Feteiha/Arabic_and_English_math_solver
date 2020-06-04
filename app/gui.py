from tkinter import *
from PIL import Image, ImageDraw
import time
from tkinter import filedialog
from tkinter import messagebox
import tkinter
from shutil import copy
import os
from app.helpers import calculate
from config import BASE_PATH



def GUI():

    width = 800
    height = 500
    center = height//2
    white = (255, 255, 255)
    green = (0,128,0)

    root = Tk()
    def getmodel():
        if var.get()=='English':
            model= os.path.join(BASE_PATH, r"train/en_model.h5")
        elif var.get()=='Arabic':
            model= os.path.join(BASE_PATH, r"train/ar_model.h5")
        return model
    
        

        
    def save_as_png(canvas,fileName):
        # save postscipt image
        filename=os.path.join(BASE_PATH, (r"app/Images/"+fileName))
        canvas.postscript(file = filename + '.eps') 
        # use PIL to convert to PNG 
        img = Image.open(filename + '.eps') 
        img.save(filename + '.png', 'png')
        ImagePath=filename+'.png'
        return ImagePath
    def submitcallback():
        ImagePath= save_as_png(cv,'Image')
        model=getmodel()
        try:
            Result,message=calculate(ImagePath,model,digital=True)
        except:
            message='An error happened, Please try again'
        if message:
            messagebox.showerror("Error", message)
        else:
            messagebox.showinfo(title='Operation Done', message='The result of the operation is '+str(Result))



    ftypes = [('JPEG files', '*.jpeg'),('PNGs files', '*.png'), ('jpg files', '*.jpg'), ('All files', '*')]
    

    def paint( event ):
       python_green = "#476042"
       x1, y1 = ( event.x - 1 ), ( event.y - 1 )
       x2, y2 = ( event.x + 1 ), ( event.y + 1 )
       cv.create_oval( x1, y1, x2, y2, fill = python_green,width=12 )
    def delete_paint(event):
        cv.delete(ALL)
    def fileDialog():
            filename_uploaded = filedialog.askopenfilename( title = "Select An cropped Image", filetype =ftypes,initialdir="/")
            dst=os.path.join(BASE_PATH, r"app/Images/Image.png")
            ImagePath=copy(filename_uploaded,dst)
            model=getmodel()
            try:
                Result,message=calculate(ImagePath,model,digital=False)
            except:
                message='An error happened, Please try again'
            if message:
                messagebox.showerror("Error", message)
            else:
                messagebox.showinfo(title='Operation Done', message='The result of the operation is '+str(Result))
            

    def fileDialog2():
            filename_uploaded = filedialog.askopenfilename(title = "Select An uncropped Image", filetype =ftypes,initialdir="/")
            dst=os.path.join(BASE_PATH, r"app/Images/Image.png")
            ImagePath=copy(filename_uploaded,dst)
            model=getmodel()
            try:
                Result,message=calculate(ImagePath,model,digital=False,scan=True)
            except:
                message='An error happened, Please try again'
            if message:
                messagebox.showerror("Error", message)
            else:
                messagebox.showinfo(title='Operation Done', message='The result of the operation is '+str(Result))

    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack(expand = YES, fill = BOTH)



    image1 = Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)


    cv.bind( "<B1-Motion>", paint )
    cv.bind( "<Button-3>",delete_paint)

    Language_lbl=Label(root, text="Choose Language of Image before submitting")

    Language_lbl.pack()
    var = StringVar(root)
    var.set("English") # initial value

    option = OptionMenu(root, var, "English","Arabic")
    option.pack()





    bottomframe = Frame(root)
    bottomframe.pack( side = BOTTOM )

    button = Button(bottomframe,text="submit the above Sketch",command = submitcallback)

    button.pack(side=tkinter.RIGHT)
    button2=Button(bottomframe, text = "upload a cropped photo",command = fileDialog)

    button2.pack(side=tkinter.RIGHT)
    button3 = Button(bottomframe,text="upload a whole page Photo",command = fileDialog2)

    button3.pack(side=tkinter.RIGHT)

    root.mainloop()
