import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

EXCEL_FILE="Lab Session Data.xlsx"

def a1():
    xls=pd.ExcelFile(EXCEL_FILE)
    df=pd.read_excel(xls,"Purchase Data")
    X=df.iloc[:,0:3].values
    y=df.iloc[:,3].values
    print("A1")
    print("Dimension:",X.shape[1])
    print("Vectors:",X.shape[0])
    print("Rank:",np.linalg.matrix_rank(X))
    print("Product Costs:",np.linalg.pinv(X)@y)

def a3():
    xls=pd.ExcelFile(EXCEL_FILE)
    df=pd.read_excel(xls,"IRCTC Stock Price")
    print("\nA3")
    print("Mean:",df.iloc[:,3].mean())
    print("Variance:",df.iloc[:,3].var())
    plt.scatter(df.iloc[:,0],df.iloc[:,8])
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Change % vs Day")
    plt.savefig("scatter.png")
    plt.close()

def a4():
    xls=pd.ExcelFile(EXCEL_FILE)
    df=pd.read_excel(xls,"thyroid0387_UCI")
    print("\nA4")
    print(df.dtypes)
    print(df.describe(include="all"))
    print(df.isnull().sum())

def a5():
    print("\nA5: Compute Jaccard and SMC on first two binary observation vectors.")

def a6():
    print("\nA6: Compute cosine similarity between first two observations.")

def a7():
    print("\nA7: Generate JC, SMC and Cosine heatmaps for first 20 observations.")

def a8():
    print("\nA8: Impute numeric columns with mean/median and categorical with mode.")

def a9():
    print("\nA9")
    print("Normalize numeric attributes using MinMaxScaler.")

def main():
    a1()
    a3()
    a4()
    a5()
    a6()
    a7()
    a8()
    a9()

if __name__=="__main__":
    main()
