import pandas as pd

"""df = pd.read_csv('data.csv')

print(df) 
print(pd.options.display.max_rows) """


df = pd.read_json('data.json')

print(df.tail(2))
print(df.info())