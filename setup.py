import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="regim",
    version="0.8.0",
    author="Shital Shah",
    author_email="sytelus@gmail.com",
    description="PyTorch Train Test Regiment Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/regim",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
          'torch', 'torchvision', 'tensorwatch', 'tensorboardX', 'numpy', 'tensorwatch'
    ]
)