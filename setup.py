from setuptools import find_packages, setup


if __name__ == '__main__':
    with open("version.txt", encoding="utf-8") as f:
        version = f.read().strip()

    with open("requirements.txt", encoding="utf-8") as f:
        requirements = f.read().strip().split()

    setup(
        name='Emotion Recognition',
        version=version,
        url="https://github.com/mark-polokhov/Emotion-Recognition",
        packages=find_packages(),
        install_requires=requirements,
    )