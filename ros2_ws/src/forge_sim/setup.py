from setuptools import setup

package_name = "forge_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Forge Team",
    maintainer_email="forge@example.com",
    description="Forge MuJoCo simulation node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "simulator_node = forge_sim.simulator_node:main",
        ],
    },
)
