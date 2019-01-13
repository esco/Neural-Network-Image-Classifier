import argparse
import numpy as np

from device import get_device
from image_utils import get_dataloaders, get_image_labels
from network import make_network, save_checkpoint, test_model, train_network 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Image directory")
    parser.add_argument("--checkpoint-name", dest="checkpoint_name",
        type=str, default="checkpoint.pth", help="Name for saved trained model")
    parser.add_argument("--arch", dest="arch", default="vgg16",
        help="Pretrained model architecture")
    parser.add_argument("--learning-rate", dest="learnrate", type=float, default=0.001,
        help="Learning rate for training")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.2,
        help="Dropout rate for training")
    parser.add_argument("--hidden-units", dest="hidden_units", type=int, default=5000,
        help="Number of hidden units for network")
    parser.add_argument("--output-size", dest="output_size", type=int, default=None,
        help="Number of outputs for network")
    parser.add_argument("--epochs", dest="epochs", type=int, default=3,
        help="Number of epochs to train network for")
    parser.add_argument("--device", dest="device", default="cuda", help="Device to train with")
    parser.add_argument("--category-names", default=None, dest="category_names",
        type=str, help="JSON mapping of images to name")
    return parser.parse_args()

def main():
    args = get_args()
    normalize_means = np.array([0.485, 0.456, 0.406])
    normalize_stds = np.array([0.229, 0.224, 0.225])
    device = get_device(args.device)
    dataloaders, image_datasets = get_dataloaders(normalize_means, normalize_stds,
        data_dir=args.dir)
    image_to_name = get_image_labels(args.category_names) if args.category_names else None
    output_size = args.output_size or len(image_to_name)
    model = make_network(hidden_size=args.hidden_units, output_size=output_size,
        dropout=args.dropout, output_dim=1, model_architecture=args.arch)
    criterion, optimizer = train_network(model, dataloaders['train'], learnrate=args.learnrate,
            device=device, epochs=args.epochs)
    print("-- testing model --")
    test_model(model, dataloaders['test'], device=device)
    print("-- saving checkpoint --")
    save_checkpoint(args.checkpoint_name, model, args.arch, image_datasets['train'].class_to_idx,
        optimizer, criterion)

if __name__ == "__main__":
    main()