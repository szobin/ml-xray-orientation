import logging
import argparse
from recognizer.classifier import Classifier
from recognizer.settings import IMAGE_DIR
import os

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='X-ray image or folder to recognize')
parser.add_argument('--img', type=str, help='The image file or folder to check', default="val")
parser.add_argument('--epoch', default=200, type=int, help='number of training epoch')

args = parser.parse_args()


def main():
    image_arg = os.path.join(IMAGE_DIR, args.img)
    logger.debug("Xray image view side classifier: {}".format(image_arg))

    classifier = Classifier(n_epoch=args.epoch)
    p = classifier.predict(image_arg)
    if p is not None:
        classifier.print_prediction(p)


if __name__ == '__main__':
    main()
