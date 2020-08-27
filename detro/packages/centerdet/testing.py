import wpcv
import detro.utils as ut
import cv2
import os, shutil, glob
import numpy as np

def test(cfg):
    ut.set_default_font_path(cfg.FONT_PATH)
    detector = cfg.get_detector()
    num_params=4
    det_type = 'bbox'
    dir=cfg.TEST_DIR
    saver = wpcv.ImageSaver(dir + '_out', remake_dir=True, auto_make_subdir=True)
    fs = glob.glob(dir + '/*.bmp')
    fs.sort()
    font = ut.get_default_font(32)
    for i, f in enumerate(fs):
        img = cv2.imread(f)
        pim = wpcv.pilimg(img)
        polygons = detector.predict(img)
        if not len(polygons):
            continue
        polys = polygons[:, :num_params].astype(np.int)
        # polys=list(filter(lambda poly:poly[num_params]>0.1,))
        im = pim
        for j, poly in enumerate(polys):
            score = polygons[j][num_params]
            label = '%.2f' % score
            if det_type == 'bbox':
                im = wpcv.draw_boxes_with_label(im, [(poly, label)], line_width=2, box_color='red', font=font,
                                                text_color='blue', offset=(0, -18))
            elif det_type=='circle':
               pass
            elif False:
                poly = np.array(poly).reshape((-1, 2))
                box=wpcv.bounding_rect(poly)
                im = wpcv.draw_boxes_with_label(im, [(box, label)], line_width=2, box_color='red', font=font,
                                                text_color='blue', offset=(0, -18))
            else:
                assert det_type == 'polygon'
                poly = np.array(poly).reshape((-1, 2))
                poly = ut.organize_polygon_points(poly)
                im = wpcv.draw_polygon(im, poly, color='red', width=4, label=label, label_xy=(0, -18),
                                       label_color='red', font=font)
                if hasattr(detector,'get_middle_results'):
                    res=detector.get_middle_results()
                    saver.save(res['center_heatmap_img'])
                    saver.save(res['corner_heatmap_img'])

        saver.save(im)
        # im=visualize(pim,polygons)
        # saver.save(im)
        print(i, f, len(polygons))

