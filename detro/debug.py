from wpcv import *
import wpcv


def heatmap2image(heatmap):
    heatmap = np.array(heatmap) * 255
    heatmap = heatmap.astype(np.uint8)
    return heatmap
_saver=None
def get_default_saver():
    global _saver
    if _saver is None:
        _saver = ImageSaver('debug/default', remake_dir=True, auto_make_subdir=True)
    return _saver
def save(im):
    saver=get_default_saver()
    return saver.save(im)