import pandas as pd

df = pd.read_csv('./samples/sample_output.txt', sep='\t', header=None)

start = 0
final = 10
print("%f,%f,%f,%f,%f,%f" %(df.iloc[start: final, :][3].mean(),
                            df.iloc[start: final, :][3].std(),
                            df.iloc[start: final, :][4].mean(),
                            df.iloc[start: final, :][4].std(),
                            df.iloc[start: final, :][5].mean(),
                            df.iloc[start: final, :][5].std()))

start = 10
final = 20
print("%f,%f,%f,%f,%f,%f" %(df.iloc[start: final, :][3].mean(),
                            df.iloc[start: final, :][3].std(),
                            df.iloc[start: final, :][4].mean(),
                            df.iloc[start: final, :][4].std(),
                            df.iloc[start: final, :][5].mean(),
                            df.iloc[start: final, :][5].std()))
