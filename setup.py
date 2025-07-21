from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="bank-rec-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    author="Roger Bovell",
    author_email="roger.bovell@gmail.com",
    description="Bank reconciliation tool with AI capabilities",
    keywords="banking, reconciliation, AI",
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "bank-rec=app.ui:main",  # Assuming ui.py has a main() function
        ],
    },
)