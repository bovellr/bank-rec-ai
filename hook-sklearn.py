# PyInstaller hook for scikit-learn
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all sklearn submodules
hiddenimports = collect_submodules('sklearn')

# Add specific sklearn modules that are commonly missed
hiddenimports += [
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._utils',
    'sklearn.utils',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.utils._typedefs',
    'sklearn.metrics',
    'sklearn.preprocessing',
    'sklearn.model_selection',
]

# Collect sklearn data files
datas = collect_data_files('sklearn') 