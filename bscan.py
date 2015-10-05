#!/usr/bin/env python

from optparse import OptionParser
import os
import cv2
import numpy as np
from scipy import ndimage
import img2pdf



def isBlack(img):
    """Return True if an image is mostly black."""
    rows, cols, ch = img.shape
    sumb = sumt = 0
    div = 10
    for y in range(rows/div, rows, rows/div):
        for x in range(cols/div, cols, cols/div):
            rgb = img[y, x]
            if rgb[0] < 100 and rgb[1] < 100 and rgb [2] < 100:
                sumb += 1
            sumt += 1
    return 1.0 * sumb / sumt > 0.9


def findPageContour(img):
    """Return the contour of the scanned page within the image.
    The first point is the one at top left. Order is counter clockwise."""
    # Find all the contours in the image
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # The longest contour is the one surrounding the page
    lengths = [len(cnt) for cnt in contours]
    longest = max(lengths)
    indexLongest = lengths.index(longest)
    cnt = contours[indexLongest]
    # Let's approximate the longest contour to have only four points
    epsilon = 0.1*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # Let's sort the points to have them starting from topleft, counter-clockwise
    v = [i[0][0]+i[0][1] for i in approx]
    topleft = v.index(min(v))
    v = [i[0][0]-i[0][1] for i in approx]
    bottomleft = v.index(min(v))
    v = [i[0][0]+i[0][1] for i in approx]
    bottomright = v.index(max(v))
    v = [i[0][0]-i[0][1] for i in approx]
    topright = v.index(max(v))
    ordered_indexes = [topleft, bottomleft, bottomright, topright]
    ordered_contour = [approx[i] for i in ordered_indexes]
    return ordered_contour


def computeA4subarea(img):
    """Return the dimension (rows, cols) of the biggest subarea
    of the image that would have the same proportion than an
    A4 page."""
    rows, cols, ch = img.shape
    rowsA4, colsA4 = 297, 210
    newrows, newcols = rows, cols
    if 1.0*rows/cols > rowsA4/colsA4:
        newrows = int(1.0 * cols * rowsA4 / colsA4)
    else:
        newcols = int(1.0 * rows * colsA4 / rowsA4)
    return newrows, newcols


def changePerspective(img, contour, newcontour):
    """Change the perspective."""
    rows, cols, ch = img.shape
    pts_origin = np.float32(contour)
    pts_destination = np.float32(newcontour)
    M = cv2.getPerspectiveTransform(pts_origin, pts_destination)
    return cv2.warpPerspective(img, M, (cols,rows))



#
# Commands
#

def splitCommand(options, args):
    """Isolate the pictures up to a black one in a sub folder."""
    path = "."
    files = os.listdir(path)
    # Determine name of sub folder
    folder = ""
    i = 0
    while True:
        folder = "document_{:0>2}".format(i)
        if folder not in files:
            break
        i += 1
    # List pictures to isolate
    print "Searching next black image: "
    imgfiles = [f for f in files if f[-4:] in [".jpg", ".JPG"]]
    index = 0
    for idx, imgfile in enumerate(imgfiles):
        print imgfile
        img = cv2.imread(os.path.join(path, imgfile))
        index = idx
        if isBlack(img):
            break
    # Move files
    if index != 0:
        imgtomove = imgfiles[:index]
        print "\nThese files will be moved to {}: {}".format(folder, " ".join(imgtomove))
        if not options.dryrun:
            os.mkdir(folder)
            for f in imgtomove:
                print os.path.join(path, f), os.path.join(path, folder, f)
                os.rename(os.path.join(path, f),
                          os.path.join(path, folder, f))
    else:
        print "\nThis file will be removed to {}".format(imgfiles[0])
        if not options.dryrun:
            os.remove(os.path.join(path, imgfiles[0]))


def rotateCommand(options, args):
    """Rotate all the images in the folder."""
    angle = 0
    if options.rotate == "t": angle = 0
    if options.rotate == "r": angle = 90
    if options.rotate == "l": angle = -90
    if options.rotate == "b": angle = 180
    path = "."
    # If no arg, select all the files of the directory
    files = args
    if len(args) == 0:
        files = os.listdir(path)
    # Filter out all the files that are not jgp
    imgfiles = [f for f in files if f[-4:] in [".jpg", ".JPG"]]
    # Rotate all the images
    for imgfile in imgfiles:
        print "Rotating ", imgfile, "by an angle of", angle
        if not options.dryrun:
            img = cv2.imread(os.path.join(path, imgfile))
            img = ndimage.rotate(img, angle)
            cv2.imwrite(os.path.join(path, imgfile), img)


def reframeCommand(options, args):
    """Reframe images on the scanned page."""
    path = "."
    # If no arg, select all the files of the directory
    files = args
    if len(args) == 0:
        files = os.listdir(path)
    # Filter out all the files that are not jgp
    imgfiles = [f for f in files if f[-4:] in [".jpg", ".JPG"]]
    # Reframe all the images
    for imgfile in imgfiles:
        print "Reframing ", imgfile
        if not options.dryrun:
            img = cv2.imread(os.path.join(path, imgfile))
            contour = findPageContour(img)
            a4rows, a4cols = computeA4subarea(img)
            newcontour = [[0,0], [0,a4rows], [a4cols,a4rows], [a4cols,0]]
            img = changePerspective(img, contour, newcontour)
            img = img[0:a4rows, 0:a4cols]
            cv2.imwrite(os.path.join(path, imgfile), img)


def generatePdfCommand(options, args):
    """Build a PDF doc with the images."""
    path = "."
    # If no arg, select all the files of the directory
    files = args
    if len(args) == 0:
        files = os.listdir(path)
    # Filter out all the files that are not jpg
    imgfiles = [f for f in files if f[-4:] in [".jpg", ".JPG"]]
    # Generate the PDF doc
    imgfiles = [os.path.join(path, i) for i in imgfiles]
    pdf_bytes = img2pdf.convert(imgfiles, dpi=25)
    file = open("out.pdf","wb")
    file.write(pdf_bytes)
    file.close()


#
# Main entry point
#


VERSION = '0.1'
USAGE = """
%prog command [options]

version=%prog """ + VERSION


def parseOptions():
    """Parse options"""
    parser = OptionParser(usage=USAGE)
    parser.add_option("-i", "--isolate",
                      action="store_true", dest="split", default=False,
                      help="Isolate first bunch of images in a sub folder.")
    parser.add_option("-g", "--generate",
                      action="store_true", dest="generate", default=False,
                      help="Generate PDF doc.")
    parser.add_option("-f", "--reframe",
                      action="store_true", dest="reframe", default=False,
                      help="Reframe the images.")
    parser.add_option("-r", "--rotate",
                      action="store", dest="rotate", default=None,
                      help="Rotate the images in the given direction.")
    parser.add_option("-d", "--dry-run",
                      action="store_true", dest="dryrun", default=False,
                      help="Dry-run: no change done.")
    aOption, aArgs = parser.parse_args()
    return (aOption, aArgs)


def main():
    """Main entry point"""
    options, args = parseOptions()
    if options.split:
        splitCommand(options, args)
    elif options.rotate is not None:
        rotateCommand(options, args)
    elif options.reframe:
        reframeCommand(options, args)
    elif options.generate:
        generatePdfCommand(options, args)
    else:
        print "Please choose a command."


if __name__ == '__main__':
    main()
