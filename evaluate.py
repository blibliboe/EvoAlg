import os
import shutil

import duckdb
import numpy as np
import pandas as pd
import pygmo as pg
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="notebook", style="whitegrid")

from tqdm import tqdm

# This file mostly serves as an inspiration for how you can work with
# the experiment results - you probably want to modify this to analyze
# what you are interested in and possibly use jupyter notebooks instead...

RESULT_DIRS = ["results"]
PREPROCESSING_DIR = "preprocessed"
PLOT_DIR = "plots"

def preprocess(input_dirs: list[str] = RESULT_DIRS, output_dir: str = PREPROCESSING_DIR, clean: bool = False):
    """Performs preprocessing for futher analysis of the raw results, to speed up further computations."""
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        if clean:
            shutil.rmtree(output_dir)
        else:
            return
    
    print("Preprocessing...")
    with duckdb.connect(":memory:") as conn:
        conn.sql(f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_csv({str([f'{d}/**/*.csv' for d in input_dirs])}, union_by_name=true)")
        conn.sql(f"COPY results TO '{output_dir}' (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (problem, method), OVERWRITE_OR_IGNORE 1)")
    print("done.")

def plot_convergence_graphs(
        y_variables = ["r2_test"],
        x_variables = ["generation", "evaluations", "time_seconds"],
        num_steps: int = 100,
        input_dir: str = PREPROCESSING_DIR,
        output_dir: str = PLOT_DIR,
        dpi: int = 600
):
    """Plots convergence graphs for each of the `x_variables` and `y_variables` specified with a resolution of `num_steps`.
    
    Note: this plot is mostly just to provide a possible starting point, for use in the report it likely is too much information in one figure.
    """
    with duckdb.connect(":memory:") as conn:
        conn.sql(f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{input_dir}/**/*.parquet', hive_partitioning = true)")

        methods = sorted([m for m,*_ in conn.sql("SELECT method FROM results GROUP BY ALL ORDER BY method ASC").fetchall()])
        problems = sorted([p for p,*_ in conn.sql("SELECT problem FROM results GROUP BY ALL ORDER BY problem ASC").fetchall()])
        problems = [problems[1], problems[4], problems[0], problems[2], problems[3]]
        
        for y_var in y_variables:
            progress = tqdm(desc=f"Plotting {y_var}...", total=len(x_variables) * len(problems) * num_steps)

            nrows, ncols = len(x_variables), len(problems)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(3 * ncols, 3 * nrows),
                gridspec_kw=dict(
                    wspace=0.4,
                    hspace=0.3,
                ),
                squeeze=False
            )
            
            hues = sns.color_palette(n_colors=len(methods))
            palette = { m:hues[i] for i,m in enumerate(methods) }
            for i, x_var in enumerate(x_variables):
                for j, problem in enumerate(problems):
                    ax = axes[i,j]
                    
                    df = pd.DataFrame()
                    max_x_value = conn.execute(f"SELECT MAX({x_var}::DOUBLE) FROM results WHERE problem = $1", [problem]).fetchone()[0]
                    for x in np.linspace(0, max_x_value, num_steps, endpoint=True):
                        df = pd.concat([df, conn.execute(f"""
                            SELECT
                                method,
                                format('{{}}.{{}}', fold, repeat)::DOUBLE AS run,
                                {x}::DOUBLE AS {x_var},
                                {"MAX" if "r2" in y_var else "MIN"}({y_var}::DOUBLE) AS {y_var}
                            FROM results
                            WHERE problem = $1 AND {x_var}::DOUBLE <= {x}
                            GROUP BY ALL
                        """, [problem]).df()], ignore_index=True)
                        progress.update()
                    
                    sns.lineplot(
                        df,
                        x=x_var,
                        y=y_var,
                        hue="method",
                        hue_order=methods,
                        estimator=np.median,
                        errorbar=("pi", 50),
                        err_kws=dict(lw=0),
                        # estimator=None,
                        # units="run",
                        legend=False,
                        ax=ax
                    )

                    if problem == "2.718 * x0 ** 2 + 3.141636 * x0":
                        problem = "(1)"
                    if problem == "x0 ** 3 - 0.3 * x0 ** 2 - 0.4 * x0 - 0.6":
                        problem = "(2)"
                    if problem == "0.3 * x0 * sin(2 * pi * x0)":
                        problem = "(3)"
                    if problem == "Airfoil":
                        problem = "(4)"
                    if problem == "Concrete Compressive Strength":
                        problem = "(5)"
                    ax.set_title(problem if i == 0 else "")
                    ax.set_ylabel(y_var if j == 0 else "")
                    ax.set_xlabel(x_var)

                    if "mse" in y_var:
                        ax.set_yscale("log")

            fig.legend(
                labels=methods,
                handles=[plt.plot([], [], color=palette[m])[0] for m in methods],
                ncols=len(methods),
                frameon=False,
                fancybox=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.0)
            )
            
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, f"convergence_{y_var}.pdf"), bbox_inches="tight", dpi=dpi)
            plt.close(fig)

def plot_hypervolume(
    input_dir: str = PREPROCESSING_DIR,
    output_dir: str = PLOT_DIR,
    objectives: list[str] = ("size", "mse_test"),
    ref_point: tuple[float, float] = (1.1, 1.1),
    combined_non_dominated_front: bool = False,
    dpi: int = 600
):
    """Computes the hypervolume per problem and method for the given (minimization!) objectives and reference point.
    
    Optionally combines the fronts of all method runs.
    """
    objectives = list(objectives)

    if combined_non_dominated_front:
        by = ["method"]
    else:
        by = ["method", "fold", "repeat"]

    with duckdb.connect(":memory:") as conn:
        conn.sql(f"CREATE OR REPLACE VIEW results AS SELECT * FROM read_parquet('{input_dir}/**/*.parquet', hive_partitioning = true)")

        methods = sorted([m for m,*_ in conn.sql("SELECT method FROM results GROUP BY ALL ORDER BY method ASC").fetchall()])
        problems = sorted([p for p,*_ in conn.sql("SELECT problem FROM results GROUP BY ALL ORDER BY problem ASC").fetchall()])
        
        hues = sns.color_palette(n_colors=len(methods))
        palette = { m:hues[i] for i,m in enumerate(methods) }

        progress = tqdm(desc="Plotting hypervolume ...", total=len(problems) * len(methods))

        rows = []
        for problem in problems:
            df = conn.execute("""
                WITH
                    -- NB: only using the last rows per run may not include the best solutions
                    -- encountered in that run if the method is not elititst
                    last_rows AS (
                        SELECT
                            problem,
                            method,
                            fold,
                            repeat,
                            MAX(generation::UINTEGER) AS generation
                        FROM results
                        WHERE problem = $1
                        GROUP BY ALL
                    )
                SELECT
                    *
                FROM results INNER JOIN last_rows ON (
                    results.problem    = last_rows.problem
                    AND results.method = last_rows.method
                    AND results.fold   = last_rows.fold
                    AND results.repeat = last_rows.repeat
                    AND results.generation = last_rows.generation
                )
                GROUP BY ALL
            """, [problem]).df()

            all_objective_values = np.array(df[objectives].values.tolist())

            # normalize objectives
            obj_min = np.min(all_objective_values, axis=0, where=np.isfinite(all_objective_values), initial=np.inf)
            obj_max = np.max(all_objective_values, axis=0, where=np.isfinite(all_objective_values), initial=-np.inf)
            normed = lambda o_vals: (np.minimum(o_vals, obj_max) - obj_min) / (obj_max - obj_min + 1e-8)

            for (method,*_), method_df in df.groupby(by=by):
                objective_values = normed(np.array(method_df[objectives].values.tolist()))
                hv = pg.hypervolume(objective_values)
                
                rows.append(dict(
                    problem=problem,
                    method=method,
                    hypervolume=hv.compute(ref_point)
                ))

                progress.update()
        
        df = pd.DataFrame(rows)
        
        g = sns.catplot(
            df,
            kind="bar" if combined_non_dominated_front else "box",
            x="method",
            order=methods,
            hue="method",
            palette=palette,
            y="hypervolume",
            col="problem",
            col_order=problems
        )

        os.makedirs(output_dir, exist_ok=True)
        g.savefig(
            os.path.join(
                output_dir,
                f"hypervolume_{'combined_' if combined_non_dominated_front else ''}" \
                    + f"{'_'.join(objectives)}_{'_'.join(map(str, list(ref_point)))}.pdf"
            ),
            bbox_inches="tight",
            dpi=dpi
        )
        plt.close(g.figure)

if __name__ == "__main__":
    preprocess(clean=True)
    plot_convergence_graphs(y_variables=["mse_train", "r2_test"])
    plot_hypervolume()
