from __future__ import annotations

import os
from pathlib import Path

pytest_plugins = ("pytester",)


#用于测试一个端到端的pytest指令是否能正常完成执行所有任务。
def test_pytest_cli_runs_pipeline(pytester, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    nova_platform_root = repo_root / "nova-platform"
    output_root = pytester.path / "nova-lite-out"

    prev_path = os.environ.get("PYTHONPATH")
    path_value = str(nova_platform_root)
    if prev_path:
        path_value = f"{path_value}:{prev_path}"
    monkeypatch.setenv("PYTHONPATH", path_value)
    result = pytester.runpytest_subprocess(
        "-p",
        "nova_platform.pytest_nova_lite",
        "--run-nova-lite",
        "--nova-lite-config",
        str(nova_platform_root / "config" / "libra_1DIE_3.2TB_24SIP_256OST.yaml"),
        "--nova-lite-shape",
        "1,16,64,64",
        "--nova-lite-dtype",
        "fp16",
        "--nova-lite-bench-version",
        "5",
        "--nova-lite-output",
        str(output_root),
        "--nova-lite-force-rerun",
    )

    assert result.ret == 0
    result.stdout.fnmatch_lines(
        [
            "*Running nova-lite pipeline*",
            "*Nova-lite report:*",
            "*Nova-lite perfetto trace:*",
        ]
    )
    assert (output_root / "gemm_v5_fp16_1-16-64-64" / "gcu00" / "report.yaml").exists()
