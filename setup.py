from setuptools import setup

setup(
    name='ccmf',
    version='0.0.1',
    python_requires='>=3.5',
    packages=['ccmf', 'ccmf.gui', 'ccmf.gui.tk', 'ccmf.gui.tk.gui_circuit', 'ccmf.ccmf', 'ccmf.model',
              'ccmf.model.base', 'ccmf.model.linear', 'ccmf.circuit', 'ccmf.dataset', 'ccmf.inference'],
    url='https://github.com/kclamar/ccmf',
    license='MIT license',
    author='Ka Chung Lam',
    author_email='kclamar@connect.ust.hk',
    description='Circuit-constrained matrix factorization',
    install_requires=['networkx', 'pandas', 'numpy', 'pyro-ppl', 'scikit-learn', 'tqdm', 'torch']
)
