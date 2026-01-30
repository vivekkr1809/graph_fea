from setuptools import setup, find_packages

setup(
    name="grafea_phasefield",
    version="0.1.0",
    description="Edge-based Phase-Field GraFEA Framework for Fracture Mechanics",
    author="GraFEA Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "matplotlib>=3.4",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "meshio>=5.0"],
    },
)
