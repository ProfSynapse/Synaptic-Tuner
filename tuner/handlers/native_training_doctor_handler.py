"""Read-only ``doctor native-training`` CLI handler."""

import json

from tuner.handlers.native_training_observation import collect_native_training_observation


class NativeTrainingDoctorHandler:
    def __init__(self, args, context):
        self.args = args

    def handle(self) -> int:
        if getattr(self.args, "doctor_fix", False) or getattr(self.args, "env_file", None) is not None:
            error = ("--fix is not supported by doctor native-training." if getattr(self.args, "doctor_fix", False)
                     else "--env-file is not supported by doctor native-training.")
            if getattr(self.args, "json", False):
                print(json.dumps({"error": error, "code": "INVALID_ARGS"}))
            else:
                print(f"Error: {error}")
            return 2
        report = collect_native_training_observation()
        if getattr(self.args, "json", False):
            print(json.dumps(report, sort_keys=True))
        else:
            print("Native training diagnostic (candidate only; no capability qualification)")
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0
