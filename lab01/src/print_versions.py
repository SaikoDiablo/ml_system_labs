import platform
from importlib.metadata import PackageNotFoundError, version


def print_versions():
    """Print Python and package versions for reproducibility."""

    print("=== ENVIRONMENT INFORMATION ===")
    print("Python:", platform.python_version())

    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "torchinfo",
        "thop",
        "onnx",
        "onnxruntime",
        "mlflow",
        "memory-profiler",
        "psutil",
        "codecarbon",
        "fastapi",
        "uvicorn",
        "pytest",
        "httpx",
        "locust",
        "requests",
        "pyarrow",
        "joblib",
        "tqdm",
    ]

    print("\n=== PACKAGE VERSIONS ===")

    for package in packages:
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: NOT INSTALLED")


if __name__ == "__main__":
    print_versions()