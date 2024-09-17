# Coding for "Interest Rate Dynamics and Commodity Prices"  

### Authors: Christophe Gouel, Qingyin Ma, and John Stachurski 


The code is written in Python (version 3.10) and has been tested on Linux (Debian 12) and Windows 11. 
It is also expected to work on macOS, though not tested.

## Folder/File Organization ##

    code/                   Folder with the source code
	output/
	|- figures/             Folder where Python writes the figures
	|- simulation_results/  Folder where Python saves intermediate results
	|- tables/              Folder where Python writes the tables

## Data, Programs, and Results ##

### Data Source ###

The only data used in the paper are interest rates and CPI from
[FRED](https://fredaccount.stlouisfed.org). They are not stored in this archive, instead, 
they are automatically downloaded when executing the file `code/real_rate.ipynb`. 

An [API key](https://fred.stlouisfed.org/docs/api/api_key.html) is required to 
enable automatic data download from [FRED](https://fredaccount.stlouisfed.org). This key
should be saved to the environment variable `FRED_API_KEY` to successfully run the file
`code/real_rate.ipynb`.

### Programs that Generates the Figures ###

| Program                          | Figure | File (folder `output/figures`) |
|----------------------------------|--------|--------------------------------|
| `code/real_rate.ipynb`           | 1      | `real_rate.pdf`                |
| Produced with Visio without code | 2      | `eq_pi.png`                    |
| `code/plot_stability.ipynb`      | 3      | `stab1.pdf` and `stab2.pdf`    |
| `code/irf_param.ipynb`           | 4      | `irf_param_Xpast_mean.pdf`     |
| `code/irf_state.ipynb`           | 5      | `irf_state_Xpast_base.pdf`     |
| `code/irf_demand_param.ipynb`    | 6      | `irf_demand_param.pdf`         |

### Tables ###

| Program             | Table | File (folder `output/tables`)           |
|---------------------|-------|-----------------------------------------|
| `code/precision.py` | E.1   | `lee_nk.tex` and `lee_delta_lambda.tex` |

### Other Programs ###

| Program             | Description          |
|---------------------|-------|
| `code/pricing.py` | The key optimality package (speculative channel) |
| `code/pricing_demang.py`| The key optimality package (global demand channel) |

Moreover, all files in the folder `code/tnr` comes from [John Burkardt's website](https://people.math.sc.edu/Burkardt/py_src/truncated_normal_rule/truncated_normal_rule.html), 
and used for computing a quadrature rule for a truncated normal distribution function. 
These files are licensed under the GNU Lesser General Public License (file `code/tnr/LICENSE`).

The calculations are run in parallel using Numba. To control the number of threads
used adjust the lines
``` python
set_num_threads(X)
```
in `code/pricing.py` and `code/pricing_demand.py`. By default, Numba uses all
available threads.

## License ##

Except when noted otherwise, the entirety of this repository is licensed under a
3-Clause BSD License (file `LICENSE`), which allows reuse with attribution.
