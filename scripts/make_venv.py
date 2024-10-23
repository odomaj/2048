import subprocess


def make_venv():
    subprocess.run(["python", "-m", "venv", "../venv"])


def install_packages():
    subprocess.run(["pip", "install", "numpy"])


if __name__ == "__main__":
    subprocess.run(["python", "-m", "venv", "../venv"])
    subprocess.run(["../venv/Scripts/activate.bat"])
