import re
from importlib import metadata
from pathlib import Path

import yaml


def _canonicalize_name(name: str) -> str:
    """Normalize distribution names per PEP 503 style."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extract_package_name(spec: str) -> str:
    """
    Extract base package name from a dependency specifier.
    Examples:
      scipy==1.11.4 -> scipy
      torch_harmonics==0.8.0 -> torch_harmonics
    """
    return re.split(r"[<>=!~ ]", spec, maxsplit=1)[0].strip()


def _packages_from_environment_file(env_path: Path) -> list[str]:
    data = yaml.safe_load(env_path.read_text())
    deps = data.get("dependencies", [])

    packages: list[str] = []

    for dep in deps:
        if isinstance(dep, str):
            packages.append(_extract_package_name(dep))
        elif isinstance(dep, dict) and "pip" in dep:
            for pip_dep in dep["pip"]:
                if isinstance(pip_dep, str):
                    packages.append(_extract_package_name(pip_dep))

    # Non-Python/runtime-only entries that won't appear as Python distributions.
    excluded = {"python", "pip", "libstdcxx-ng"}
    return [pkg for pkg in packages if pkg and pkg not in excluded]


def test_environment_platform_packages_are_installed():
    """
    Ensure all Python packages declared in env.yml
    are present in the active Python environment.
    """
    repo_root = Path(__file__).resolve().parents[1]
    env_file = repo_root / "env.yml"
    assert env_file.exists(), f"Missing environment file: {env_file}"

    declared_packages = _packages_from_environment_file(env_file)
    assert declared_packages, "No packages were parsed from env.yml"

    installed = {_canonicalize_name(dist.metadata["Name"]) for dist in metadata.distributions()}

    # Common alias: conda package name -> Python distribution name.
    alias_map = {"pytorch": "torch"}

    missing = []
    for pkg in declared_packages:
        candidate = alias_map.get(pkg, pkg)
        if _canonicalize_name(candidate) not in installed:
            missing.append(pkg)

    assert not missing, (
        "The following packages are declared in env.yml but "
        f"not installed in this environment: {missing}"
    )
