from PIL import Image, ImageDraw, ImageFont
from colour import Color
import numpy as np
import requests
im = Image.open(requests.get('https://picsum.photos/512', stream=True).raw)
#im.show()


font = ImageFont.load_default()
letter_width = font.getsize("x")[0]
letter_height = font.getsize("x")[1]

print(letter_width, letter_height)

# font = ImageFont.load_default()
# (left, top, right, bottom) = font.getbbox("x")

# height = font.getlength("M", direction='ttb')

# #letter_width = size[0]
# #letter_height = size[1]

# print(left, top, right, bottom, height)
# #print(size)