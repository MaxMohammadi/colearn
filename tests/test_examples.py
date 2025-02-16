# ------------------------------------------------------------------------------
#
#   Copyright 2021 Fetch.AI Limited
#
#   Licensed under the Creative Commons Attribution-NonCommercial International
#   License, Version 4.0 (the "License"); you may not use this file except in
#   compliance with the License. You may obtain a copy of the License at
#
#       http://creativecommons.org/licenses/by-nc/4.0/legalcode
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
import os
import signal
import subprocess
from pathlib import Path
from typing import List, Dict, Sequence

import pytest

REPO_ROOT = Path(__file__).absolute().parent.parent
EXAMPLES_DIR = REPO_ROOT / "colearn_examples" / "ml_interface"
GRPC_EXAMPLES_DIR = REPO_ROOT / "colearn_examples" / "grpc"

GITHUB_ACTION = bool(os.getenv("GITHUB_ACTION", ""))

if GITHUB_ACTION:
    COLEARN_DATA_DIR = Path("/pvc-data/")
    TFDS_DATA_DIR = str(COLEARN_DATA_DIR / "tensorflow_datasets")
    PYTORCH_DATA_DIR = str(COLEARN_DATA_DIR / "pytorch_datasets")

else:
    COLEARN_DATA_DIR = Path(
        os.getenv("COLEARN_DATA_DIR",
                  os.path.expanduser(os.path.join('~', 'datasets'))))

    TFDS_DATA_DIR = os.getenv("TFDS_DATA_DIR",
                              str(os.path.expanduser(os.path.join('~', "tensorflow_datasets"))))
    PYTORCH_DATA_DIR = os.getenv("PYTORCH_DATA_DIR",
                                 str(os.path.expanduser(os.path.join('~', "pytorch_datasets"))))

FRAUD_DATA_DIR = COLEARN_DATA_DIR / "ieee-fraud-detection"
XRAY_DATA_DIR = COLEARN_DATA_DIR / "chest_xray"
COVID_DATA_DIR = COLEARN_DATA_DIR / "covid"

NUMBER_OF_LEARNERS = 3
STANDARD_DEMO_ARGS: List[str] = ["-p", "1", "-n", str(NUMBER_OF_LEARNERS)]

EXAMPLES_WITH_KWARGS = [
    (EXAMPLES_DIR / "keras_cifar.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),  # script 0
    (EXAMPLES_DIR / "keras_fraud.py", [FRAUD_DATA_DIR], {}),
    (EXAMPLES_DIR / "keras_mnist.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    (EXAMPLES_DIR / "keras_mnist_diffpriv.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    (EXAMPLES_DIR / "keras_xray.py", [XRAY_DATA_DIR], {}),
    (EXAMPLES_DIR / "mli_fraud.py", [FRAUD_DATA_DIR], {}),
    (EXAMPLES_DIR / "mli_random_forest_iris.py", [], {}),
    (EXAMPLES_DIR / "pytorch_cifar.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    (EXAMPLES_DIR / "pytorch_covid.py", [COVID_DATA_DIR], {}),
    (EXAMPLES_DIR / "pytorch_mnist.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    (EXAMPLES_DIR / "pytorch_mnist_diffpriv.py", [], {"PYTORCH_DATA_DIR": PYTORCH_DATA_DIR}),
    (EXAMPLES_DIR / "pytorch_xray.py", [XRAY_DATA_DIR], {}),
    (EXAMPLES_DIR / "run_demo.py", ["-m", "PYTORCH_XRAY", "-d", str(XRAY_DATA_DIR / "train")] + STANDARD_DEMO_ARGS, {}),
    (EXAMPLES_DIR / "run_demo.py", ["-m", "KERAS_MNIST"] + STANDARD_DEMO_ARGS, {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    (EXAMPLES_DIR / "run_demo.py", ["-m", "KERAS_CIFAR10"] + STANDARD_DEMO_ARGS, {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    (EXAMPLES_DIR / "run_demo.py", ["-m", "PYTORCH_COVID_XRAY", "-d", str(COVID_DATA_DIR)] + STANDARD_DEMO_ARGS, {}),
    (EXAMPLES_DIR / "run_demo.py", ["-m", "FRAUD", "-d", str(FRAUD_DATA_DIR)] + STANDARD_DEMO_ARGS, {}),
    (EXAMPLES_DIR / "xgb_reg_boston.py", [], {}),
    (GRPC_EXAMPLES_DIR / "mlifactory_grpc_mnist.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
    (GRPC_EXAMPLES_DIR / "mnist_grpc.py", [], {"TFDS_DATA_DIR": TFDS_DATA_DIR}),
]

IGNORED: List[str] = []

print("Killing old servers that might have been left from incomplete termination")
subprocess.run("pkill -e -f grpc", shell=True, check=False)
print("Done killing")


@pytest.mark.parametrize("script,cmd_line,test_env", EXAMPLES_WITH_KWARGS)
@pytest.mark.slow
def test_a_colearn_example(script: str, cmd_line: List[str], test_env: Dict[str, str]):
    env = os.environ
    env["MPLBACKEND"] = "agg"  # disable interactive plotting
    env["COLEARN_EXAMPLES_TEST"] = "1"  # enables test mode, which sets n_rounds=1
    env.update(test_env)
    print("Additional envvars:", test_env)

    if script in IGNORED:
        pytest.skip(f"Example {script} marked as IGNORED")

    full_cmd: Sequence = ["python", str(script)] + cmd_line
    print("Full command", full_cmd)
    subprocess.run(full_cmd,
                   env=env,
                   timeout=20 * 60,
                   check=True
                   )


def test_all_examples_included():
    examples_list = {EXAMPLES_DIR / x.name for x in EXAMPLES_DIR.glob('*.py')}
    assert examples_list <= {x[0] for x in EXAMPLES_WITH_KWARGS}


# NB: this test depends on the existence of /tmp/mnist/0 etc. which are created in the previous tests.
# This means that this test must run after those ones.
GRPC_EXAMPLES_WITH_KWARGS: List = [
    (GRPC_EXAMPLES_DIR / "run_grpc_demo.py", ["-m", "KERAS_MNIST",
                                              "-d", "KERAS_MNIST",
                                              "-l", "/tmp/mnist/0,/tmp/mnist/1,/tmp/mnist/2",
                                              "-n", str(NUMBER_OF_LEARNERS), "--n_rounds", "1"], {}),
]


@pytest.mark.parametrize("script,cmd_line,test_env", GRPC_EXAMPLES_WITH_KWARGS)
@pytest.mark.slow
def test_a_colearn_grpc_example(script: str, cmd_line: List[str], test_env: Dict[str, str]):
    env = os.environ
    env["MPLBACKEND"] = "agg"  # disable interactive plotting
    env.update(test_env)
    print("Additional envvars:", test_env)

    grpc_servers = subprocess.Popen(["python", str(REPO_ROOT / "colearn_grpc" / "scripts" / "run_n_grpc_servers.py"),
                                     "-n", str(NUMBER_OF_LEARNERS)])

    full_cmd: Sequence = ["python", str(script)] + cmd_line
    print("Full command", " ".join(full_cmd))
    subprocess.run(full_cmd,
                   env=env,
                   timeout=20 * 60,
                   check=True
                   )
    grpc_servers.send_signal(signal.SIGINT)
    grpc_servers.wait()
