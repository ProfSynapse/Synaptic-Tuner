"""Copy the two reviewed root catalogs into the installed resource namespace."""

from pathlib import Path

from setuptools.command.build_py import build_py


CATALOGS = ("cli-first-tool-schemas.json", "Tools/tool_schemas.json")


class PublicAssetBuild(build_py):
    def run(self):
        super().run()
        source_root = Path(__file__).resolve().parent
        for member in CATALOGS:
            destination = Path(self.build_lib) / "synaptic_tuner" / "runtime" / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(source_root / member), str(destination))

    def get_outputs(self, include_bytecode=1):
        return [*super().get_outputs(include_bytecode), *(
            str(Path(self.build_lib) / "synaptic_tuner" / "runtime" / member)
            for member in CATALOGS
        )]
