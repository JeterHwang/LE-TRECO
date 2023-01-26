from argparse import ArgumentParser, Namespace
from pathlib import Path
import subprocess
from subprocess import PIPE

def runcmd(command):
    bash_command = command.split()
    ret = subprocess.run(bash_command, stdout=PIPE, stderr=PIPE)
    if ret.returncode == 0:
        return ret.stdout
    else:
        print("Error !!")
        return ret.stderr

def delete(args):
    for path in args.data_path.iterdir():
        if path.is_dir():
            folder_name = path.stem
            sep = folder_name.find('_')
            index = folder_name[:sep]
            name = folder_name[sep+1:]
            for file in path.iterdir():
                if file.suffix == '.cm' or file.stem == index or file.stem == name or 'unaligned' in file.stem:
                    continue
                runcmd(f"rm {file.absolute().resolve()}")
                
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Directory to the dataset.",
        default="./data/blast",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    delete(args)
    