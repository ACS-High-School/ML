import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="b3o-fedlearn",
    version="1.0.0",
    author="skswjdekdud",
    author_email="skswjdekdud1103@gmail.com",
    description="a project for federated learning using AWS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ACS-High-School",
    project_urls={
        "Bug Tracker": "https://github.com/ACS-High-School/ML/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
