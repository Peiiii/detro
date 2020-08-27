import numpy as np
import math


def normalize(image):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	mean = np.array([[mean]])
	std = np.array([[std]])
	image = (image.astype(np.float32) - mean) / std
	return image


def denormalize(image):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	mean = np.array([[mean]])
	std = np.array([[std]])
	image = (image.astype(np.float32) * std) + mean
	return image


def dist_point_line(p, line):
	x, y = p
	((x0, y0), (x1, y1)) = line

	if x0 == x1:
		return abs(x - x0)
	else:
		line = np.polyfit([x0, x1], [y0, y1], 1)
		k, b = line
		return abs(k * x - y + b) / math.sqrt(k ** 2 + 1)


def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y


def min_max_vertical_dist(points, line):
	k, b = line
	vdists = points[:, 1] - (k * points[:, 0] + b)
	return min(vdists), max(vdists)


def kb2two_point(line):
	k, b = line
	p1, p2 = [0, b], [10, k * 10 + b]
	return p1, p2


def get_split_line_of_two_quad(quad1, quad2):
	p0, p1 = quad1[1:3]
	p2, p3 = quad2[0], quad2[3]
	p02 = np.mean([p0, p2], axis=0)
	p13 = np.mean([p1, p3], axis=0)
	return [p02, p13]


def approx_polygon(quads):
	q1 = quads[0]
	q2 = quads[-1]
	# p0=np.mean([q1[0],q1[3]],axis=0)
	# p1=np.mean([q2[1],q2[2]],axis=0)

	points = np.array(quads).reshape((-1, 2))
	line = np.polyfit(points[:, 0], points[:, 1], 1)
	k, b = line
	dmin, dmax = min_max_vertical_dist(points, line)
	line0 = kb2two_point([k, b + dmax])
	line2 = kb2two_point([k, b + dmin])
	line3 = [q1[0], q1[3]]
	line1 = [q2[1], q2[2]]

	p0 = line_intersection(line0, line3)
	p1 = line_intersection(line0, line1)
	p2 = line_intersection(line1, line2)
	p3 = line_intersection(line2, line3)
	big_quad = np.array([p0, p1, p2, p3])
	split_points = []
	for i in range(len(quads) - 1):
		l_quad, r_quad = quads[i], quads[i + 1]
		line = get_split_line_of_two_quad(l_quad, r_quad)
		p0 = line_intersection(line, line0)
		p1 = line_intersection(line, line2)
		split_points.append([p0, p1])
	# center=big_quad.mean(axis=0)
	# split_offsets=np.zeros((17*3))
	# quad_offsets=np.zeros()

	res = dict(
		split_points=split_points,
		big_quad=big_quad,
	)
	return res
