import pandas as pd
import os
import matplotlib.pyplot as plt


def extract():
    files = [
        os.path.join(RAW_DATA_FILE_PATH, f) for f in os.listdir(RAW_DATA_FILE_PATH)
    ]

    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(
            f, encoding="latin-1", usecols=COLS, skipfooter=2, engine="python"
        )  # HWIN adds info on final line we want to skip
        df["Series"] = os.path.basename(f)
        dfs.append(df)

    panel = pd.concat(dfs, ignore_index=True)
    return panel


def clean(panel: pd.DataFrame):
    panel["Datetime"] = pd.to_datetime(
        panel["Date"] + " " + panel["Time"], dayfirst=True
    )
    panel = panel.drop(["Date", "Time"], axis=1)
    return panel


def visualize(panel: pd.DataFrame):
    # print(panel.groupby("Series").describe().loc["eldenring"].transpose())

    _, ax = plt.subplots(3, 3, sharey=True)
    for i, (name, df) in enumerate(panel.groupby("Series")):
        row, col = i // 3, i % 3  # map flat index to 2D grid position
        df[["Total CPU Usage [%]", "GPU Utilization [%]"]].rolling(
            window=30
        ).mean().plot(ax=ax[row, col])
        ax[row, col].set_title(name)
    plt.show()


def main():
    panel = extract()
    cleaned = clean(panel)
    visualize(cleaned)


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    main()
