import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summarize(df):
    shape = df.shape
    info = df.info()
    describe = df.describe()
    distributions = df.hist(figsize=(24, 10), bins=20)
    #pairplot = sns.pairplot(df)
    return shape, info, describe, distributions, #pairplot
