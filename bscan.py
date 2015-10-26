#!/usr/bin/env python

from optparse import OptionParser
import os
import cv2
import numpy as np
from scipy import ndimage
import img2pdf
import math


def isBlack(img):
    """Return True if the image is mostly black."""
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


def findContourWithLonguestSegment(contours):
    """Return the contour which contain the longest segment."""
    maxi = 0
    index = -1
    for i, contour in enumerate(contours):
        # Compute segment lengths and find longest one
        c = contour.tolist()
        seglens = [math.sqrt((x2-x1)**2+(y2-y1)**2)
                   for ([[[x1, y1]], [[x2, y2]]])
                   in zip(c,
                          c[1:]+c[:1])]
        longuest = max(seglens)
        # Save index of this contour if it has the longest segment
        if longuest > maxi:
            maxi = longuest
            index = i
    return contours[index]


def findPageContour(img):
    """Return the contour of the scanned page within the image.
    The first point is the one at top left. Order is counter clockwise."""
    # Find all the contours in the image
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # Approximate the contours (and filter out the ones that do not have 4 edges)
    appContours = [cv2.approxPolyDP(cnt,
                                    0.1*cv2.arcLength(cnt, True),
                                    True)
                   for cnt in contours
                   if len(cnt) >= 4]
    appContours = [cnt for cnt in appContours if len(cnt) >= 4]
    # The contour surrounding the page is the one with the longest segment
    cnt = findContourWithLonguestSegment(appContours)
    # Let's sort the points to have them counter-clockwise, starting from topleft
    v = [i[0][0]+i[0][1] for i in cnt]
    topleft = v.index(min(v))
    v = [i[0][0]-i[0][1] for i in cnt]
    bottomleft = v.index(min(v))
    v = [i[0][0]+i[0][1] for i in cnt]
    bottomright = v.index(max(v))
    v = [i[0][0]-i[0][1] for i in cnt]
    topright = v.index(max(v))
    ordered_indexes = [topleft, bottomleft, bottomright, topright]
    ordered_contour = np.array([cnt[i] for i in ordered_indexes])
    return ordered_contour


def rotateAboutCenter(src, angle, scale=1.):
    """Rotate an image by a given angle."""
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat,
                          (int(math.ceil(nw)), int(math.ceil(nh))),
                         flags=cv2.INTER_LANCZOS4)


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


def inputFiles(options, args):
    """Select files to consider based on command lines arguments."""
    # Files are given in args
    files = args
    # ... or from the current directory
    if len(args) == 0:
        files = [os.path.join(".", f) for f in os.listdir(".")]
    # Filter out all the files that are not jgp images
    imgfiles = [f for f in files if f[-4:] in [".jpg", ".JPG"]]
    return imgfiles


#
# Commands
#

def splitCommand(options, args):
    """Move the pictures up to a black one in a sub folder."""
    imgfiles = inputFiles(options, args)
    # Determine name of sub folder
    folder = ""
    i = 0
    while True:
        folder = "document_{:0>2}".format(i)
        if folder not in os.listdir("."):
            break
        i += 1
    # List pictures to isolate
    print "Searching next black image: "
    index = 0
    for idx, imgfile in enumerate(imgfiles):
        print imgfile
        img = cv2.imread(imgfile)
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
                print f, os.path.join(".", folder, os.path.basename(f))
                os.rename(f,
                          os.path.join(".", folder, os.path.basename(f)))
    else:
        print "\nThis 'black' file will be removed to {}".format(imgfiles[0])
        if not options.dryrun:
            os.remove(imgfiles[0])


def rotateCommand(options, args):
    """Rotate all the images in the folder."""
    angle = 0
    if options.rotate == "t": angle = 0
    if options.rotate == "r": angle = 90
    if options.rotate == "l": angle = -90
    if options.rotate == "b": angle = 180
    imgfiles = inputFiles(options, args)
    # Rotate all the images
    for imgfile in imgfiles:
        print "Rotating ", imgfile, "by an angle of", angle
        if not options.dryrun:
            img = cv2.imread(imgfile)
            img = rotateAboutCenter(img, angle)
            cv2.imwrite(imgfile, img)


def reframeCommand(options, args):
    """Reframe the image on the scanned page."""
    imgfiles = inputFiles(options, args)
    # Reframe all the images
    for imgfile in imgfiles:
        print "Reframing ", imgfile
        if not options.dryrun:
            img = cv2.imread(imgfile)
            contour = findPageContour(img)
            a4rows, a4cols = computeA4subarea(img)
            newcontour = [[0,0], [0,a4rows], [a4cols,a4rows], [a4cols,0]]
            img = changePerspective(img, contour, newcontour)
            img = img[0:a4rows, 0:a4cols]
            cv2.imwrite(imgfile, img)


def contrastCommand(options, args):
    """Improve the contrast of the image."""
    imgfiles = inputFiles(options, args)
    # Reframe all the images
    for imgfile in imgfiles:
        print "improving contrast of ", imgfile
        if not options.dryrun:
            img = cv2.imread(imgfile, 0)
            #cv2.equalizeHist(img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            cv2.imwrite(imgfile, img)


def generatePdfCommand(options, args):
    """Build a PDF doc with the images."""
    imgfiles = inputFiles(options, args)
    # Generate the PDF doc
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
    parser.add_option("-r", "--rotate",
                      action="store", dest="rotate", default=None,
                      help="Rotate the images in the given direction.")
    parser.add_option("-f", "--reframe",
                      action="store_true", dest="reframe", default=False,
                      help="Reframe the images.")
    parser.add_option("-c", "--contrast",
                      action="store_true", dest="contrast", default=False,
                      help="Improve images contrast.")
    parser.add_option("-g", "--generate",
                      action="store_true", dest="generate", default=False,
                      help="Generate PDF doc.")
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
    elif options.contrast:
        contrastCommand(options, args)
    elif options.reframe:
        reframeCommand(options, args)
    elif options.generate:
        generatePdfCommand(options, args)
    else:
        print "Please choose a command."


if __name__ == '__main__':
    main()
