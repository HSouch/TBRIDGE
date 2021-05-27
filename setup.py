import setuptools


with open("README.md", "r") as readme:
    long_description = readme.read()


setuptools.setup(
    name="tbridge",
    version="1.4.7",
    licemse="BSD 3-clause 'New' or 'Revised' license",
    author="Harrison Souchereau",
    author_email="harrison.souchereau@yale.edu",
    description="Testing BRIghtness Deviations in Galaxy profile Extractions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hsouch/tbridge",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-clause 'New' or 'Revised' license",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
