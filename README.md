# SARS-CoV-2 Wastewater Variant Explorer

Set up the Python environment with `uv sync` and `source .venv/bin/activate`, then run `quarto preview`.

Only `wastewater-variant-explorer.qmd` and `acknowledgements.qmd` are published. The root URL and legacy `dashboard1.html` URL redirect to `wastewater-variant-explorer.html`, which retains the scatter plot, variant data browser, and project metadata browser. The former Home, About, and second dashboard pages have been removed.

Run `quarto render` to rebuild the complete site before publishing; rendering a single page does not clean retired pages from an existing output directory. Rendering the explorer also rewrites `data/transformed_variant_plotting_data.tsv`.
