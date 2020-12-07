import setuptools


with open("README.md", "r") as readme:
    long_description = readme.read()


setuptools.setup(
    name="tbridge",
    version="1.2.0",
    author="Harrison Souchereau",
    author_email="harrison.souchereau@yale.edu",
    description="Testing BRIghtness Deviations in Galaxy profile Extractions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hsouch/tbridge",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
