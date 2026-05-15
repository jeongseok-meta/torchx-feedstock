import subprocess

import torchx
from torchx import specs
from torchx.components import utils


def test_appdef_specs():
    role = specs.Role(
        name="trainer",
        image="conda-forge/torchx-test",
        entrypoint="python",
        args=["-m", "train", "--rank", specs.macros.replica_id],
        env={"APP_ID": specs.macros.app_id},
        num_replicas=2,
        resource=specs.Resource(cpu=1, gpu=0, memMB=256),
    )
    app = specs.AppDef(name="demo", roles=[role])

    values = specs.macros.Values(
        img_root="/image",
        app_id="demo-app",
        replica_id="1",
        rank0_env="MASTER_ADDR",
    )
    resolved = values.apply(app.roles[0])

    assert torchx.__version__ == "0.7.0"
    assert app.name == "demo"
    assert resolved.args[-1] == "1"
    assert resolved.env["APP_ID"] == "demo-app"
    assert resolved.resource.cpu == 1


def test_builtin_component_and_cli():
    app = utils.echo("hello conda-forge", image="busybox", num_replicas=3)
    role = app.roles[0]

    assert app.name == "echo"
    assert role.entrypoint == "echo"
    assert role.args == ["hello conda-forge"]
    assert role.num_replicas == 3

    result = subprocess.run(
        ["torchx", "run", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout


if __name__ == "__main__":
    test_appdef_specs()
    test_builtin_component_and_cli()
