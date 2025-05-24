import sklearn
import tensorflow as tf
import pandas as pd
import streamlit as st
import joblib
import numpy
import matplotlib
import scipy
import re
import pickle

def print_version(pkg, name=None):
    name = name or pkg.__name__
    version = getattr(pkg, '__version__', None)
    if version is None:
        print(f"{name}: No version attribute found")
    else:
        print(f"{name} version: {version}")

print_version(sklearn)
print_version(tf, 'tensorflow')
print_version(pd, 'pandas')
print_version(st, 'streamlit')
print_version(joblib)
print_version(numpy)
print_version(matplotlib, 'matplotlib')
print_version(scipy)

print("re: built-in module, no version info")
print("pickle: built-in module, no version info")
