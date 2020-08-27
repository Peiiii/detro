import cv2
from PIL import Image,ImageDraw

def draw_circle(img,center,radius,fillcolor='red',text=None,font=None,text_color=None):
    x,y=center=(int(center[0]),int(center[1]))
    r=radius
    draw = ImageDraw.Draw(img)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=None,outline=fillcolor)
    if text:
        text_color=text_color or fillcolor
        draw.text(center,text,font=font,fill=text_color)
    return img