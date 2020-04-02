
import numpy as np
from skimage import io
import argparse


def bright(img, gain):
    return np.clip(img*1.+gain, 0, 255)


def contrast(img, gain):
    return np.clip((img-127.)*gain+127., 0, 255)


def gamma(img, gain):
    return np.power(img/255, 1/gain)*255


def main():
    parser = argparse.ArgumentParser(description='A Simple Point Processing Tool implemented by Numpy')
    parser.add_argument('-i', dest='input', help='Input image file path', type=str)
    parser.add_argument('-o', dest='output', help='Output file path', type=str, default='./Pictures/Output.png')
    parser.add_argument('-b', dest='brightness', help='Change the brightness of image', type=float)
    parser.add_argument('-c', dest='contrast', help='Change the contrast of image', type=float)
    parser.add_argument('-g', dest='gamma', help='Apply gamma transformation for image', type=float)
    parser.add_argument('-e', dest='equalization', help='Apply histogram equalization for image', action='store_true')
    parser.add_argument('-m', dest='matching',
                        help='Apply histogram matching for image. Please input another image', type=str)
    args = parser.parse_args()
    # print(args)
    if args.input:
        img = io.imread(args.input)
    else:
        print('Please input an image!')
        return
    if args.brightness:
        print('Changing brightness...')
        io.imsave(args.output, bright(img, args.brightness))
    if args.contrast:
        print('Changing contrast...')
        io.imsave(args.output, contrast(img, args.contrast))
    if args.gamma:
        print('Applying gamma transformation... ')
        io.imsave(args.output, gamma(img, args.gamma))
    if args.equalization:
        print('Applying histogram equalization...')
        return
    if args.matching:
        print('Matching histogram with another image...')
        return


if __name__ == '__main__':
    main()

