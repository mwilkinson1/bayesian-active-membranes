import pandas as pd

# Read the CSV file into a DataFrame
NE_df = pd.read_csv("NE Strain Combined.csv", header=None)
LE_df = pd.read_csv("LE Strain Combined.csv", header=None)
parameter_df = pd.read_csv("Parameters.csv", header=None)

def string_to_list(input_string):
    cleaned_string = input_string.strip('[]')
    numbers_str = cleaned_string.split(',')
    numbers_list = [float(num) for num in numbers_str]
    return numbers_list

def convertComponentXX(df_column):
    temp_df = df_column.str.strip("[]").str.split(",")
    XX = temp_df.str.get(0).astype('double')
    return XX

def convertComponentYY(df_column):
    temp_df = df_column.str.strip("[]").str.split(",")
    YY = temp_df.str.get(1).astype('double')
    return YY

LE_XX = LE_df.apply(convertComponentXX, axis=1)
LE_YY = LE_df.apply(convertComponentYY, axis=1)

LE_XX_df = pd.DataFrame(LE_XX)
LE_YY_df = pd.DataFrame(LE_YY)

LE_XX_df.to_csv("LE_XX.csv", header=False, index=False)
LE_YY_df.to_csv("LE_YY.csv", header=False, index=False)
