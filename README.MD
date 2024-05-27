# Suggestions

- Conventional commits: see https://www.conventionalcommits.org/en/v1.0.0/
- Branch development: use git branches to organize your work. Each task is a branch with a PR.


My version of public learning is creating this notebook of difficulties and what was done to overcome it. Hopefully, it can be useful to improve the course as well

# Module 1
## 1 - Local Env Setup (macOs)

1.1. Check [Local MacOS envprep by Adetbekov](https://github.com/adetbekov/mlops-zoomcamp/blob/main/01-intro/macos-local-envprep.md) for Docker setup

1.2. Package Version Control using Poetry

- Install poetry: `pip install poetry`
- Start environemnt: `poetry init` which will create a `pyproject.toml` file with main dependencies. Feel free to skip dependency and install them later using `poetry add`
- Start a .venv `poetry shell` and add .venv to `.gitignore`

(Bonus)
[Updating code that uses seaborn.distplot](https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751)


# Module 2
## Adding MLFlow
- Poetry already showed to be useful since if we use pip to install MLFlow it will install 2.1.3 by default which is incompatible with pyarrow 16
- Also, when using requirements.txt in the videos, make sure to define versions of packages. Example: my first version of sklearn was not supported with MLFlow's autologging


## First test of running train.py with MLFlow autlogging

- ![First test of running train.py with MLFlow autlogging](image.png)


- ![Hyperoptimization in MLFlow](image-1.png)


- ![Final experiment with test set, saving the best model](best_model_saved.png)