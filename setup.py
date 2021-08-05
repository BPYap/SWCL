import os
import setuptools

setuptools.setup(
    name="retinal_fundus_encoder",
    version="0.1",
    packages=["retinal_fundus_encoder"]
)

if not os.path.exists("SupContrast/__init__.py"):
    open("SupContrast/__init__.py", 'a').close()

setuptools.setup(
    name="SupContrast",
    packages=["SupContrast"]
)

setuptools.setup(
    name="big_transfer",
    packages=["big_transfer", "big_transfer.bit_pytorch"]
)