import sys
import math
import random
import os
from PIL import Image, ImageDraw

# tweakable constants for transformations
# Variations in position of the symbol,
# variations in the orientation of the symbol,
# variations in the size of the symbol,
# variations in the thickness of the strokes in the symbol,
# variations in the size and number of stray marks such as ellipsoids drawn in the image.
# The control of the amount of these kinds of variations should be pulled out into tweakable constants at the top of the zener_generator.py file.

# scaling factors to be applied by multiplication of the allowable range
from numpy.random.mtrand import randint

transformation_parameters = {
    "position": 1,
    "size": 2,
    "thickness": 1,
    "orientation": 1.5,
    "stray_marks": {"size": 1, "number": 2, "enable": True}
}

# constants
global_minimum_size = 6
g_max_thickness = 3
canvas_size = 25
available_symbols = ['O', 'P', 'Q', 'S', 'W']
orientation_range = {"high": 90, "low": 0}
drawDoodle = (True, False)  # Should there be doodles on the canvas or not
available_doodles = ['ellipse', 'line']


def create_card(canvas, dp):
    draw_symbol = {
        "O": create_circle,
        "P": create_plus,
        "W": create_waves,
        "Q": create_square,
        "S": create_star
    }

    im_returned = draw_symbol[dp["symbol"]](canvas)

    # out_im = im_returned.rotate(dp["orientation"])
    # return out_im
    return im_returned


def create_boundingboxes(minimum_size=global_minimum_size):
    # outer Bounding Box
    x1 = random.randint(1, canvas_size / 2)
    y1 = random.randint(1, canvas_size / 2)

    max_width = canvas_size - x1
    max_height = canvas_size - y1

    side_length = random.randint(minimum_size, min(max_width, max_height))  # side length of the square bounding box

    x2 = x1 + side_length
    y2 = y1 + side_length

    # bounding box of circle
    outer_boundingbox = (x1, y1, x2, y2)


    # inner Bounding Box
    skew = transformation_parameters['thickness'] * random.randint(1, g_max_thickness)
    x1p = x1 + skew
    x2p = x2 - skew
    y1p = y1 + skew
    y2p = y2 - skew

    inner_boundingbox = (x1p, y1p, x2p, y2p)


    return outer_boundingbox, inner_boundingbox


# fill two ellipses in with coordinates defined by inner and outer bounding boxes
def create_circle(canvas):
    draw = ImageDraw.Draw(canvas)
    outer, inner = create_boundingboxes()

    draw.ellipse(outer, fill="black")  # draw outer circle
    draw.ellipse(inner, fill="white")  # draw inner circle

    return canvas


# fill two bounding boxes in
def create_square(canvas):
    draw = ImageDraw.Draw(canvas)
    outer, inner = create_boundingboxes()

    draw.rectangle(outer, fill="black")  # draw outer circle
    draw.rectangle(inner, fill="white")  # draw inner circle

    return canvas


# create 2 intersecting lines that seperate the bounding box into 4 equal parts
def create_plus(canvas):
    draw = ImageDraw.Draw(canvas)
    outer, _junk = create_boundingboxes()
    line_width = random.randint(1, g_max_thickness) * transformation_parameters['thickness']

    yleft = (outer[3] - outer[1]) / 2

    left_arm_coordinate = (outer[0], outer[1] + yleft)  # x1, y2-y1/2
    right_arm_coordinate = (outer[2], outer[1] + yleft)  # x2, y2-y1/2

    horizontal_line = (left_arm_coordinate, right_arm_coordinate)

    xtop = (outer[2] - outer[0]) / 2
    head_coordinate = (outer[0] + xtop, outer[1])  # ,y1
    tail_coordinate = (outer[0] + xtop, outer[3])

    vertical_line = (head_coordinate, tail_coordinate)


    draw.line(horizontal_line, fill="black", width=line_width)
    draw.line(vertical_line, fill="black", width=line_width)

    return canvas


# draw 5 lines that correspond to points on the bounding box
def create_star(canvas):
    draw = ImageDraw.Draw(canvas)
    outer, _junk = create_boundingboxes()
    side_length = outer[2] - outer[0]
    line_width = random.randint(1, g_max_thickness) * transformation_parameters['thickness']

    # crossbar - top horizontal line
    xc_left = outer[0]
    xc_right = outer[2]
    yc = outer[1] + (outer[3] - outer[1]) / 3.0  # y1 + 1/3(y2-y1)

    crossbar_left = (xc_left, yc)
    crossbar_right = (xc_right, yc)
    crossbar = (crossbar_left, crossbar_right)
    draw.line(crossbar, fill="Black", width=line_width)

    # upper left diagonal
    uld_x = outer[0] + (outer[2] - outer[0]) / 2.0
    uld_begin = (uld_x, outer[1])
    uld_end = (outer[0], outer[3])
    upper_left_diagonal = (uld_begin, uld_end)
    draw.line(upper_left_diagonal, fill="Black", width=line_width)

    # upper right diagonal
    urd_begin = uld_begin
    urd_end = (outer[2], outer[3])
    upper_right_diagonal = (urd_begin, urd_end)
    draw.line(upper_right_diagonal, fill="Black", width=line_width)

    # lower left diagonal
    lld_begin = uld_end
    lld_end = crossbar_right
    lower_left_diagonal = (lld_begin, lld_end)
    draw.line(lower_left_diagonal, fill="Black", width=line_width)

    # lower right diagonal
    lrd_begin = urd_end
    lrd_end = crossbar_left
    lower_right_diagonal = (lrd_begin, lrd_end)
    draw.line(lower_right_diagonal, fill="Black", width=line_width)

    # draw.rectangle(outer)
    return canvas


def create_waves(canvas):
    draw = ImageDraw.Draw(canvas)
    outer, _junk = create_boundingboxes(12)

    side_length = outer[2] - outer[0]
    offset = side_length / 4.0
    amplitude = random.randint(1, 2)
    line_width = random.randint(1, 2) * transformation_parameters['thickness']
    # line_width = 1
    x_coords = range(outer[0], outer[2])
    y_starting_point = outer[1] + (outer[3] - outer[1]) / 2.0

    def apply_sine(input):
        x = input
        # x = 1/6.0 * math.pi * input / side_length
        return amplitude * math.sin(x)

    y = map(apply_sine, x_coords)

    for i in range(0, len(x_coords) - 1):
        center_wave = ((x_coords[i], y_starting_point + y[i]), (x_coords[i + 1], y_starting_point + y[i + 1]))
        draw.line(center_wave, fill="Black", width=line_width)

        top_wave = ((x_coords[i], y_starting_point + y[i] - offset), (x_coords[i + 1], y_starting_point + y[i + 1] -
                                                                      offset))
        draw.line(top_wave, fill="Black", width=line_width)

        bottom_wave = ((x_coords[i], y_starting_point + y[i] + offset), (x_coords[i + 1], y_starting_point + y[i + 1] +
                                                                         offset))
        draw.line(bottom_wave, fill="Black", width=line_width)
        # draw.point((x_coords[i], y_starting_point + y[i]))
    return canvas.rotate(90)

# add noise to canvas
def add_stray_marks(canvas, dp):
    draw = ImageDraw.Draw(canvas)

    # generate coordinates of the marks

    # coordinates
    x1 = randint(0,canvas_size)
    y1 = randint(0,canvas_size)

    x2 = x1 + dp['stray_marks']['size']
    y2 = y1 + dp['stray_marks']['size']


    if dp['stray_marks']['shape'] == 'ellipse':
        draw.ellipse(((x1,y1),(x2,y2)),fill="black")
    else:
        draw.line(((x1,y1),(x2,y2)),fill="black")
    return canvas


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Please supply a folder name and number of training data to generate"
        exit(1)
    num_examples = int(sys.argv[2])
    folder_name = sys.argv[1]
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    drawing_parameters = {}
    for i in range(0, num_examples):

        drawing_parameters["symbol"] = random.choice(available_symbols)
        drawing_parameters["orientation"] = random.randint(orientation_range['low'], orientation_range['high']) * \
                                            transformation_parameters['orientation']

        drawing_parameters["stray_marks"] = {"shape": available_doodles[random.randint(0, 1)],
                                             "size": transformation_parameters['stray_marks']['size'],
                                             "number": transformation_parameters['stray_marks']['number'] }

        image_file = os.path.join(folder_name, str(i) + "_" + drawing_parameters["symbol"] + ".png")



        # white_canvas = Image.new("RGB", (25, 25), color="white")


        new_image = create_card(Image.new("RGB", (canvas_size, canvas_size), color="white"), drawing_parameters)

        # randomly choose to add noise to the canvas
        for i in range(0, transformation_parameters["stray_marks"]["number"]):
            if (random.choice([True, False])):
                new_image = add_stray_marks(new_image,drawing_parameters)

        im2 = new_image.convert('RGBA') # convert to have alpha layer
        rot = im2.rotate(drawing_parameters["orientation"] )
        fff = Image.new('RGBA', rot.size, (255,) * 4)

        out = Image.composite(rot, fff, rot)
        out.convert("1").save(image_file,"PNG")
