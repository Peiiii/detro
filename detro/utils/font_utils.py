from PIL import ImageFont
import os

_root = os.path.dirname(__file__)
_default_font_path = _root + '/../data/msyh.ttf'


def set_default_font_path(path):
    global _default_font_path
    _default_font_path = path


def get_default_font(fontsize=16):
    font = _default_font_path
    return ImageFont.truetype(font, size=fontsize)
