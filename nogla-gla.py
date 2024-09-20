import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

model_path = 'Tkinter-Practice_Glasses-or-No-Glasses/saved_model.h5'  
model = tf.keras.models.load_model(model_path)
# Image size for the model
IMG_SIZE = (256, 256)


def preprocess_input(image):
    image = tf.image.resize(image, IMG_SIZE)  
    image = tf.cast(image, tf.float32) 
    image = image / 255.0  
    return image

def show_image():
    global filename, imported_image
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title="Select image file",
                                          filetypes=(("JPG file", "*.jpg"),
                                                     ("PNG file", "*.png"),
                                                     ("JPEG file", "*.jpeg"),
                                                     ("All files", "*.*")))
    if filename:
        img = Image.open(filename)
        img = img.resize(IMG_SIZE, Image.LANCZOS)  
        img = np.array(img)  
        img = preprocess_input(img)  
        imported_image = ImageTk.PhotoImage(Image.fromarray((img.numpy() * 255).astype(np.uint8)))  # Convert back to PIL Image and then to Tkinter PhotoImage
        lbl.configure(image=imported_image)
        lbl.image = imported_image  
        predict_image(img)

def predict_image(img):
    img_array = tf.expand_dims(img, axis=0)  
    predictions = model.predict(img_array)
    score = predictions[0][0]
    if score >= 0.5:
        label = "With Glasses"
    else:
        label = "Without Glasses"
    #result_label.config(text=f"Prediction: {label} (Confidence: {score:.2f})")
    result_label.config(text=f"Prediction: {label}")



window = tk.Tk()
window.title("Glasses or No Glasses")
window.geometry("900x500+100+100")
window.configure(bg="White")

img_icon = Image.open("Tkinter-Practice_Glasses-or-No-Glasses/test.png")
img_icon = img_icon.resize((70, 100), Image.LANCZOS)
photo_img = ImageTk.PhotoImage(img_icon)
window.iconphoto(False, photo_img)

tk.Label(window, image=photo_img, bg="#fff").place(x=10, y=10)
tk.Label(window, text="Glasses or No Glasses", font="arial 25 bold", fg="yellow", bg="black").place(x=90, y=50)

selectimage = tk.Frame(window, width=275, height=330, bg="#d6dee5")
selectimage.place(x=10, y=120)
f = tk.Frame(selectimage, bg="black", width=256, height=256)
f.place(x=10, y=10)
lbl = tk.Label(f, bg="black")
lbl.place(x=0, y=0)
tk.Button(selectimage, text="Select image", width=12, height=2, font="arial 14 bold", command=show_image).place(x=10, y=265)

result_label = tk.Label(window, text="Prediction: ", font="arial 20 bold", bg="white")
result_label.place(x=300, y=200)

window.mainloop()
