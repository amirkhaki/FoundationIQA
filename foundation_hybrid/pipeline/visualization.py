import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_latex_table(df, group_prefix, output_dir="tables/"):
    os.makedirs(output_dir, exist_ok=True)
    # Filter df by group prefix
    group_df = df[df['study_id'].str.startswith(group_prefix)]
    if group_df.empty:
        return
        
    pivot = group_df.pivot(index='study_id', columns='dataset', values='SRCC')
    
    # Format to latex with booktabs
    latex_str = pivot.to_latex(float_format="%.4f", bold_rows=True, na_rep="-", 
                               column_format="l" + "c"*len(pivot.columns))
    
    # Highlight best in bold, second best underlined - string replacement logic
    # (Implementation details omitted for brevity, but here is where it happens)
    
    with open(os.path.join(output_dir, f"{group_prefix}_table.tex"), "w") as f:
        f.write(latex_str)

def generate_sensitivity_plot(df, param_name, study_ids, output_dir="figures/"):
    os.makedirs(output_dir, exist_ok=True)
    # Filter df
    plot_df = df[df['study_id'].isin(study_ids)]
    if plot_df.empty: return
    
    plt.figure(figsize=(8, 5))
    for dataset in plot_df['dataset'].unique():
        ds_data = plot_df[plot_df['dataset'] == dataset]
        plt.plot(ds_data['study_id'], ds_data['SRCC'], marker='o', label=dataset)
        
    plt.title(f"Sensitivity: {param_name}")
    plt.ylabel("SRCC")
    plt.xlabel(param_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sensitivity_{param_name}.pdf"))
    plt.close()

def generate_all_visualizations(csv_file="master_results.csv"):
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)
    
    generate_latex_table(df, "A1")
    generate_latex_table(df, "B1")
    generate_latex_table(df, "C1")
    
    # generate_sensitivity_plot(df, "Gate Steepness (k)", [...])
    
if __name__ == "__main__":
    # DO NOT EXECUTE - THIS IS JUST THE CODE
    # generate_all_visualizations()
    pass
