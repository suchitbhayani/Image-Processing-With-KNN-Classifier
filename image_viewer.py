"""
Courtesy of ChatGPT
"""

import tkinter as tk
from PIL import Image, ImageTk

class ImageZoomer:
    def __init__(self, image_path):
        """Initialize the ImageZoomer class"""
        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.root.geometry("500x550")  # Increased height to accommodate the label

        self.canvas = tk.Canvas(self.root, bg="black", width=500, height=500)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

        # Store the original image and a copy of the image
        self.original_image = Image.open(image_path)
        self.image_copy = self.original_image.copy()
        self.image_tk = None
        self.image_id = None

        self.display_image()

        self.canvas.bind("<Configure>", self.resize_image)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<ButtonPress-1>", self.scroll_start)
        self.canvas.bind("<B1-Motion>", self.scroll_move)
        self.canvas.bind("<Motion>", self.on_hover)

        self.color_label = tk.Label(self.root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.color_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.last_color = None  # To avoid updating label unnecessarily

    def display_image(self):
        """Display the image on the canvas"""
        self.image_copy.thumbnail((500, 500), Image.NEAREST)
        self.image_tk = ImageTk.PhotoImage(self.image_copy)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def resize_image(self, event):
        """Resize the image to fit the canvas"""
        new_width = event.width
        new_height = event.height
        self.image_copy = self.original_image.copy()
        self.image_copy.thumbnail((new_width, new_height), Image.NEAREST)
        self.image_tk = ImageTk.PhotoImage(self.image_copy)
        self.canvas.itemconfig(self.image_id, image=self.image_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def zoom(self, event):
        """Zoom the image in or out"""
        # Zooming functionality
        factor = 1.1 if event.delta > 0 else 0.9
        new_width = int(self.image_copy.width * factor)
        new_height = int(self.image_copy.height * factor)
        self.image_copy = self.original_image.copy()
        # Do not change the image if the new size is smaller than the original
        if new_width < self.image_copy.width and new_height < self.image_copy.height:
            return
        self.image_copy = self.image_copy.resize((new_width, new_height), Image.NEAREST)
        self.image_tk = ImageTk.PhotoImage(self.image_copy)
        self.canvas.itemconfig(self.image_id, image=self.image_tk)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def scroll_start(self, event):
        # Scrolling functionality
        self.canvas.scan_mark(event.x, event.y)

    def scroll_move(self, event):
        # This function is responsible for scrolling functionality
        # It moves the canvas to the position of the mouse drag
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_hover(self, event):
        """Display color value on hover"""
        # Get the canvas coordinates of the mouse pointer
        x = self.canvas.canvasx(event.x)  
        y = self.canvas.canvasy(event.y)
        try:
            # Get the current size of the image
            current_image_width = self.image_copy.width
            current_image_height = self.image_copy.height

            # Calculate the ratio of the original image size to the current image size
            # This is used to map the mouse coordinates to the image coordinates
            width_ratio = self.original_image.width / current_image_width
            height_ratio = self.original_image.height / current_image_height

            # Adjust the mouse coordinates to match the image coordinates
            image_x = int(x * width_ratio)
            image_y = int(y * height_ratio)

            # Get the color of the pixel at the mouse pointer
            pixel_color = self.original_image.getpixel((image_x, image_y))
            # Display the color of the pixel in a label
            self.color_label.config(text=f"Color: {pixel_color}")
        except:
            pass

    def run(self):
        # Start the Tkinter event loop
        self.root.mainloop()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        app = ImageZoomer(image_path)
        app.run()
    except FileNotFoundError:
        print("Image not found.")