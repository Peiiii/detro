
import xml.etree.ElementTree as ET
def read_pascalvoc_annotation(file,cls2idx):
    root  = ET.parse(open(file,'r')).getroot()

    size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)
    # path = root.find('path').text
    objects = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in cls2idx:
            continue
        cls_id = cls2idx[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymax').text))
        obj = [list(b), cls_id]
        objects.append(obj)
    return objects