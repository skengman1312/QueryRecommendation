import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    d = {"baseline": [0.73, 0.65, 0.51, 0.54],
         "QSRS(c)": [0.88, 0.83, 0.66, 0.61],
         "QSRS(c+u)": [0.82, 0.78, 0.64, 0.55],
         "QSRS(u)": [0.88, 0.82, 0.80, 0.67]

         }

    df = pd.DataFrame(d)
    df.index = df.index + 1
    print(df)
    df.plot(title="Mean expected utility", legend=True, xticks=[1, 2, 3, 4], xlabel="Instance", yerr=0.01, ylim=[0.5, 1])
    plt.savefig("res.png")
    plt.show()

    d = {"baseline": [0.78, 0.67, 0.51, 0.57],
         "QSRS(c)": [0.93, 0.85, 0.68, 0.63],
         "QSRS(c+u)": [0.94, 0.89, 0.69, 0.64],
         "QSRS(u)": [0.88, 0.82, 0.78, 0.67]

         }

    df = pd.DataFrame(d)
    df.index = df.index + 1
    print(df)
    df.plot(title="Mean expected utility with unique entries", legend=True, xticks=[1, 2, 3, 4], xlabel="Instance",
            yerr=0.01, ylim=[0.5, 1])
    plt.savefig("res_u.png")

    plt.show()
