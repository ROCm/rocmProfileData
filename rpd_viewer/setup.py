from setuptools import setup, find_packages

setup(
    name="rpd-viewer",
    version="1.0",
    description="Interactive viewer for RocmProfileData (.rpd) trace files",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "rpd_viewer": ["assets/*", "RPD_INFO.md"],
    },
    python_requires=">=3.8",
    install_requires=[
        "dash",
        "dash-ag-grid",
        "plotly",
        "pandas",
        "openai",
        "rocpd",
    ],
    entry_points={
        "console_scripts": [
            "rpd-viewer=rpd_viewer.app:main",
        ],
    },
    zip_safe=False,
)
