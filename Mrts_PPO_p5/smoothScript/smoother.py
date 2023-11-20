import pandas as pd
import matplotlib.pyplot as plt

name = "model4_winrate"

plt.switch_backend('Qt5Agg')
df = pd.read_csv(f"{name}.csv")
print(df)

TSBOARD_SMOOTHING = [0.999]

smooth = []
for ts_factor in TSBOARD_SMOOTHING:
    smooth.append(df.ewm(alpha=(1 - ts_factor)).mean())

for ptx in range(1):
    plt.subplot(1, 3, ptx + 1)
    plt.plot(df["Step"], df["Value"], alpha=0.4)
    plt.plot(smooth[ptx]["Step"], smooth[ptx]["Value"])
    plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING[ptx]))
    plt.grid(alpha=0.3)


# Save the smoothed data to separate CSV files
for ptx, ts_factor in zip(range(1), TSBOARD_SMOOTHING):
    smooth[ptx].to_csv(f"{name}_smoothed.csv".format(ts_factor), index=False)


#plt.show()


