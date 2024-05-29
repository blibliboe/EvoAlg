


if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    ground_truth = "0.3 * x0 * sin(2 * pi * x0)"
    
    mo_front = pd.read_json('plots/data/mo_data_20240529-100643.json', lines=True)
    so_front = pd.read_json('plots/data/so_data_20240529-100643.json', lines=True)
    
    fronts = pd.concat([so_front, mo_front], ignore_index=True)
    
    ax = sns.lineplot(
        mo_front,
        x="size",
        y="mse_train",
        hue="type",
        marker="o",
        alpha=0.5,
        legend="brief"
    )
    # for _, row in fronts.iterrows():
    #     ax.text(row["size"] + 0.2, row["mse_train"], row["expression"], fontsize=8)
    ax.set_yscale("log")
    ax.set_title(f"Pareto Approximation Fronts for {ground_truth}")
    plt.show()