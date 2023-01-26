import torch
from argparse import ArgumentParser, Namespace
from pathlib import Path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Directory to the dataset.",
        default="./data/blast",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Directory to the dataset.",
        default="./ckpt",
    )
    args = parser.parse_args()
    return args

def save_model(args):
    state_dict = torch.load(args.model_path, map_location=torch.device("cpu"))
    new_dict = {}
    for key, value in state_dict.items():
        if 'embedding.' in key:
            key = key.replace('embedding.', '')
        # if 'layers' in key:
        #     key = key.replace('layers', 'lstm')
        if 'layers' in key:
            layer_id = key.split('.')[1]
            postfix = key.split('.')[2].split('_')
            postfix[2] = postfix[2][0] + layer_id
            key = '_'.join(postfix)
            new_dict[key] = value
            
    model = torch.nn.LSTM(
        21,
        1024,
        3,
        batch_first=True,
        bidirectional=True
    )
    model_dict = model.state_dict()
    model_dict.update(new_dict)
    torch.save(model_dict, args.save_path / "LSTM.pt")

if __name__ == "__main__":
    args = parse_args()
    args.save_path.mkdir(parents=True, exist_ok=True)
    save_model(args)

