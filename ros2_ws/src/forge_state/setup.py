from setuptools import setup

package_name = "forge_state"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Forge Team",
    maintainer_email="forge@example.com",
    description="Forge state publisher node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "state_node = forge_state.state_node:main",
        ],
    },
)
