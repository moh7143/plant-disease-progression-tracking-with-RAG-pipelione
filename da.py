from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import numpy as np
from textwrap import wrap
import os

slides = [
    ("Step 1", "Open Figma and click New Design File."),
    ("Step 2", "Press F and choose Desktop 1440 frame."),
    ("Step 3", "Press R to create background rectangle."),
    ("Step 4", "Choose Linear Gradient and add colors."),
    ("Step 5", "Create login card using Rectangle Tool."),
    ("Step 6", "Set Corner Radius to 30 and add Blur."),
    ("Step 7", "Press T and add logo and heading."),
    ("Step 8", "Create input fields and buttons."),
    ("Step 9", "Create Booking Screen and Navbar."),
    ("Step 10", "Create movie cards and seat layout."),
    ("Step 11", "Add food cards and booking summary."),
    ("Step 12", "Use Prototype tab to connect screens.")
]

width, height = 1280, 720
frames = []

try:
    title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
    body_font = ImageFont.truetype("DejaVuSans.ttf", 32)
except:
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

for step, desc in slides:
    img = Image.new("RGB", (width, height), (18, 18, 35))
    draw = ImageDraw.Draw(img)

    draw.rectangle((40, 40, width-40, height-40), outline=(229, 9, 20), width=6)

    draw.text((80, 100), step, fill="white", font=title_font)

    wrapped = wrap(desc, width=38)
    y = 250
    for line in wrapped:
        draw.text((80, y), line, fill=(230,230,230), font=body_font)
        y += 55

    draw.text((80, 620), "Figma Cinema App Design Tutorial", fill=(255,215,0), font=body_font)

    frame = np.array(img)

    # Repeat each frame for duration
    for _ in range(30):  # ~3 seconds/frame at 10 fps
        frames.append(frame)

video_path = "/mnt/data/figma_cinema_tutorial_fixed.mp4"

writer = imageio.get_writer(
    video_path,
    fps=10,
    codec='libx264',
    format='FFMPEG'
)

for frame in frames:
    writer.append_data(frame)

writer.close()

print("Fixed video created successfully!")
print(video_path)