import argparse
import numpy as np

from device import get_device
from image_utils import get_image_labels
from network import load_checkpoint, get_prediction_with_labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Image to predict species from")
    parser.add_argument("checkpoint", type=str, help="Pretrained model checkpoint")
    parser.add_argument("--topk", dest="topk", type=int, default=5,
        help="Number of predictions to return")
    parser.add_argument("--category-names", dest="category_names", default=None,
        type=str, help="JSON mapping of images to names")
    parser.add_argument("--device", dest="device", default=None, type=str,
        help="Device to train with")
    return parser.parse_args()

def main():
    args = get_args()
    normalize_means = np.array([0.485, 0.456, 0.406])
    normalize_stds = np.array([0.229, 0.224, 0.225])
    image_to_name = get_image_labels(args.category_names) if args.category_names else None
    print("loading checkpoint")
    saved_model = load_checkpoint(args.checkpoint)
    print("getting prediciton for single image")
    image_path = args.image
    device = get_device(args.device)
    class_names, probs, normalized_probs = get_prediction_with_labels(saved_model, image_path,
        image_to_name, normalize_means, normalize_stds, topk=args.topk, device=device)
    print("top {} results: {}".format(args.topk, class_names))
    print("probabilities: ", probs)
    print("relative probabilities: ", normalized_probs)

if __name__ == "__main__":
    main()