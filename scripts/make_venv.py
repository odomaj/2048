import subprocess
import sys
from os import getcwd
from pathlib import Path


def make_venv() -> None:
    subprocess.run(["python", "-m", "venv", "../venv"])
    print("made virtual environment venv")


def install_packages() -> None:
    subprocess.run(["pip", "install", "-r", "../requirements.txt"])
    print("installed packages")


if __name__ == "__main__":
    if Path(__file__).parent.absolute() != Path(getcwd()).absolute():
        print("must be in scripts directory, exiting")
        exit(0)
    if (
        Path(sys.executable).parent.parent.parent.absolute()
        == Path(__file__).parent.parent.absolute()
    ):
        install_packages()
    else:
        make_venv()
