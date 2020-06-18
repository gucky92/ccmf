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
    install_requires=['networkx', 'pandas', 'numpy', 'pyro-ppl', 'scikit-learn', 'tqdm', 'torch'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
    download_url='https://github.com/kclamar/ccmf/archive/v0.0.1.tar.gz'
)
