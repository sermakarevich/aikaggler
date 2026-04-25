# Konwinski Prize Demo Submission 

- **Author:** Sohier Dane
- **Votes:** 616
- **Ref:** sohier/konwinski-prize-demo-submission
- **URL:** https://www.kaggle.com/code/sohier/konwinski-prize-demo-submission
- **Last run:** 2025-01-27 19:20:05.637000

---

```python
import io
import os
import shutil
import subprocess

import pandas as pd
import polars as pl

import kaggle_evaluation.konwinski_prize_inference_server
```

The evaluation API requires that you set up a server which will respond to inference requests. We have already defined the server; you just need write the predict function. When we evaluate your submission on the hidden test set the client defined in `konwinski_prize_gateway` will run in a different container with direct access to the hidden test set and hand off the data.

Your code will always have access to the published copies of the files.

```python
instance_count = None

def get_number_of_instances(num_instances: int) -> None:
    """ The very first message from the gateway will be the total number of instances to be served.
    You don't need to edit this function.
    """
    global instance_count
    instance_count = num_instances
```

```python
first_prediction = True


def predict(problem_statement: str, repo_archive: io.BytesIO, pip_packages_archive: io.BytesIO, env_setup_cmds_templates: list[str]) -> str:
    """ Replace this function with your inference code.
    Args:
        problem_statement: The text of the git issue.
        repo_path: A BytesIO buffer path with a .tar containing the codebase that must be patched. The gateway will make this directory available immediately before this function runs.
        pip_packages_archive: A BytesIO buffer path with a .tar containing the wheel files necessary for running unit tests.
        env_setup_cmds_templates: Commands necessary for installing the pip_packages_archive.
    """
    global first_prediction
    if not first_prediction:
        return None  # Skip issue.

    # Unpack the codebase to be patched into a directory that won't be exported when
    # the notebook is saved.
    archive_path = '/tmp/repo_archive.tar'
    with open(archive_path, 'wb') as f:
        f.write(repo_archive.read())
    repo_path = 'repo'
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    shutil.unpack_archive(archive_path, extract_dir=repo_path)
    os.remove(archive_path)

    """
    Unpack pip_packages if you want to run unit tests on your patch.
    Note that editing unit tests with your patch -- even to add valid tests -- can cause your submission to be flagged as a failure.
    Most of the relevant repos use pytest for running tests. You will almost certainly need to run only a subset of the unit tests to avoid running out of inference time.
    """
    pip_archive_dir = '/tmp/pip_packages_archive.tar'
    with open(pip_archive_dir, 'wb') as f:
        f.write(pip_packages_archive.read())
    pip_packages_path = '/path/to/pip_packages'
    if os.path.exists(pip_packages_path):
        shutil.rmtree(pip_packages_path)
    shutil.unpack_archive(pip_archive_dir, extract_dir=pip_packages_path)
    os.remove(pip_archive_dir)

    # Get env setup cmds by setting the pip_packages_path
    env_setup_cmds = [cmd.format(pip_packages_path=pip_packages_path) for cmd in env_setup_cmds_templates]

    # Run env setup for the repo
    subprocess.run(
        "\n".join(env_setup_cmds),
        shell=True,
        executable="/bin/bash",
        cwd=repo_path,
    )

    first_prediction = False
    # Instead of a valid diff, let's just submit a generic string. This will definitely fail.
    return "Hello World"
```

When your notebook is run on the hidden test set, inference_server.serve must be called within 15 minutes of the notebook starting or the gateway will throw an error. If you need more than 15 minutes to load your model you can do so during the very first predict call, which does not have the usual 30 minute response deadline.

```python
inference_server = kaggle_evaluation.konwinski_prize_inference_server.KPrizeInferenceServer(
    get_number_of_instances,   
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/konwinski-prize/',  # Path to the entire competition dataset
            '/kaggle/tmp/konwinski-prize/',   # Path to a scratch directory for unpacking data.a_zip.
        ),
        use_concurrency=True,  # This can safely be disabled for purposes of local testing if necessary.
    )
```