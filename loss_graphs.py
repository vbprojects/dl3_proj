#%%
import pandas as pd
#%%
df1 = pd.read_csv("csv (1).csv")
df2 = pd.read_csv("csv (2).csv")
df1['KNN Validation Accuracy'] = df1['Value']
df1.rename(columns={'Step': 'Epoch', 'Value': 'ProxyAnchor Train Loss'}, inplace=True)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load logs
acc_df = pd.read_csv("csv (1).csv")   # KNN validation accuracy
loss_df = pd.read_csv("csv (2).csv")  # ProxyAnchor train loss

# Standardize schema
acc_df = acc_df.rename(columns={"Step": "Epoch", "Value": "KNN Validation Accuracy"})
loss_df = loss_df.rename(columns={"Step": "Epoch", "Value": "ProxyAnchor Train Loss"})

# Keep only what we need and merge on epoch
plot_df = pd.merge(
    acc_df[["Epoch", "KNN Validation Accuracy"]],
    loss_df[["Epoch", "ProxyAnchor Train Loss"]],
    on="Epoch",
    how="inner",
).sort_values("Epoch")

# Paper-style settings
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

fig, ax1 = plt.subplots(figsize=(4.0, 4.0))  # good single-column-ish paper size
ax2 = ax1.twinx()

# Colorblind-friendly colors
c_acc = "#1b9e77"   # green
c_loss = "#d95f02"  # orange

# Left axis: KNN validation accuracy
sns.lineplot(
    data=plot_df, x="Epoch", y="KNN Validation Accuracy",
    ax=ax1, color=c_acc, marker="o", markersize=3, linewidth=1.8, #label="KNN Validation Accuracy"
)

# Right axis: ProxyAnchor train loss
sns.lineplot(
    data=plot_df, x="Epoch", y="ProxyAnchor Train Loss",
    ax=ax2, color=c_loss, marker="s", markersize=3, linewidth=1.8, #label="ProxyAnchor Train Loss"
)

# Axis labels and limits
ax1.set_xlabel("Epoch")
ax1.set_ylabel("KNN Validation Accuracy", color=c_acc)
ax2.set_ylabel("ProxyAnchor Train Loss", color=c_loss)
ax1.tick_params(axis="y", colors=c_acc)
ax2.tick_params(axis="y", colors=c_loss)
ax1.set_xlim(plot_df["Epoch"].min(), plot_df["Epoch"].max())
ax1.set_ylim(0.0, 1.0)  # accuracy scale
ax1.grid(True, which="major", axis="both", alpha=0.35)

# Unified legend (outside, above plot)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(
    h1 + h2, l1 + l2,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False
)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("knn_proxyanchor_dual_axis.pdf", bbox_inches="tight")  # vector format for papers
fig.savefig("knn_proxyanchor_dual_axis.png", bbox_inches="tight")
plt.show()