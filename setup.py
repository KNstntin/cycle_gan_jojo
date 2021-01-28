from setuptools import setup

setup(
    name='Cycle_Gan_tools',
    url='https://github.com/KNstn/cycle_gan_jojo',
    author='NikolaevKA',
    author_email='kanikolaev_1@edu.hse.ru',
    packages=['packages'],
    install_requires=['numpy', 'torch', 'torchvision', 'PIL'],
    version='0.1',
    license='MIT',
    description='Some tools for work with CycleGAN'
)