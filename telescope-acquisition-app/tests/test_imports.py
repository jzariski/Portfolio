def test_imports():
    import streamlit  # noqa: F401
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
    import xgboost  # noqa: F401
    import h5py  # noqa: F401
    # Local modules (relative import assumes running from repo root)
    from ModelBuilder import ModelBuilder, ModelBuilderNN  # noqa: F401
    from Parsing import TextParser  # noqa: F401
