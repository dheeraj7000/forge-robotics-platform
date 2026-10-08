from setuptools import setup

package_name = "forge_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Forge Team",
    maintainer_email="forge@example.com",
    description="Forge controller node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "controller_node = forge_control.controller_node:main",
        ],
    },
)
