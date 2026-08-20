# Network Chemotaxis

This repository contains the numerical approximation code as well as the latex write up for the work to reproduce the numerical results of:

Numerical approximation of nonhomogeneous boundary conditions on networks for a hyperbolic system of chemotaxis modeling the Physarum dynamics (2018) by Beretti, G. and Natalini, R.

## Running the Python code

The Python code lives in the `numerical_approximation` folder. It uses a tool called **uv** to manage the Python version and all dependencies, so you do not need to install Python or any packages by hand.

### 1. Install uv

**macOS or Linux** — open a terminal and run:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** — open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **Windows note:** If PowerShell blocks the command with an "execution policy" error, the command above already works around this for you (that is what `-ExecutionPolicy ByPass` does). Use PowerShell, not the older Command Prompt (`cmd.exe`), for all the commands below.

> **Using WSL instead?** If you have [WSL](https://learn.microsoft.com/windows/wsl/install) (Windows Subsystem for Linux) set up, that is a good option too. Inside your WSL terminal (e.g. Ubuntu), follow the **macOS or Linux** instructions above and below, not the PowerShell ones — WSL runs a real Linux environment, so the Linux commands apply directly. A few things to keep in mind:
> - Clone or copy this repository into your Linux home folder (e.g. `~/network-chemotaxis`) rather than working on it under `/mnt/c/...`. Files under `/mnt/c` are your Windows drive and are much slower to work with from WSL.
> - `uv`, Python, and all dependencies installed inside WSL are separate from any Windows installation of them. Run every command (`uv sync`, `uv run ...`) from the same WSL terminal.
> - `uv run jupyter lab` should open Jupyter in your default Windows browser automatically. If it does not, copy the `http://localhost:8888/...` link that is printed in the terminal and paste it into your browser manually.

After installing, close and reopen your terminal so the `uv` command is available. You can check it worked by running:

```sh
uv --version
```

### 2. Set up the project

From the repository root, move into the project folder:

```sh
cd numerical_approximation
```

Then let uv install the correct Python version (3.14) and all dependencies:

```sh
uv sync
```

This creates a local `.venv` folder with everything the project needs. You do not need to activate it yourself — the next step handles that for you.

### 3. Run the code

To run the project's command-line script:

```sh
uv run numerical-approximation
```

To open the Jupyter notebook (`heat_equation_graph.ipynb`):

```sh
uv run jupyter lab
```

This opens Jupyter in your browser. From there you can open and run the notebook.

> **Windows note:** These `uv run ...` commands work the same way in PowerShell as on macOS/Linux. Just make sure you run them from inside the `numerical_approximation` folder, using PowerShell rather than Command Prompt.
