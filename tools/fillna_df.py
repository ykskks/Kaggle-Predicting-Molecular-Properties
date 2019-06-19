import feather
import pandas as pd

FILES = ['./features/CoulombInteraction_train.feather',
        './features/CoulombInteraction_test.feather']

for f in FILES:
    df = feather.read_dataframe(f)
    df.fillna(0, inplace=True)
    df.to_feather(f)