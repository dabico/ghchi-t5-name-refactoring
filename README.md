Before even attempting to run the script, make sure you have the latest version of [`pip`](https://pypi.org/project/pip/) and [`virtualenv`](https://pypi.org/project/virtualenv/) installed:

```bash
$ python -m pip --version
$ python -m virtualenv --version
```

You can install them by running:

```bash
$ python -m pip install --user --upgrade pip
$ python -m pip install --user virtualenv
```

Running the script requires an isolated virtual environment.  
Execute the following in your shell:

```bash
$ python -m venv venv
```

A `venv` directory will be created.  
After that, activate the environment:

```bash
$ source venv/bin/activate
```

In order for the script to run, you must also download all the packages listed in `requirements.txt`:
```bash
$ pip install -U -r requirements.txt
```

After that, you can finally execute the script with the following command:
```bash
$ python ./predictor.py ./relative/path/to/methods.csv
```

Where `methods.csv` is the file containing the masked method code of a project, obtained through our platform.  
The resulting `predictions.csv` file will be created in the `out` directory.  
To leave the virtual environment, run:
```bash
$ deactivate
```
