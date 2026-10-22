"""Fixed stdlib worker for the private native guardian CPU qualification."""

from __future__ import annotations

import os
import stat
import sys


REQUEST = b'{"schema":"synaptic-native-cpu-worker-request/v1"}\n'
RESULT = b'{"schema":"synaptic-native-cpu-worker-result/v1","status":"ok"}\n'


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--request":
        return 120
    root = os.getcwd()
    if sys.argv[2] != os.path.join(root, "request.json"):
        return 120
    os.umask(0o077)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        request_fd = os.open("request.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            info = os.fstat(request_fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_size != len(REQUEST)
                    or os.read(request_fd, len(REQUEST) + 1) != REQUEST):
                return 120
        finally:
            os.close(request_fd)
        result_fd = os.open("worker-result.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600, dir_fd=root_fd)
        with os.fdopen(result_fd, "wb") as output:
            output.write(RESULT)
            output.flush()
            os.fsync(output.fileno())
        return 0
    finally:
        os.close(root_fd)


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        # The guardian counts output bytes; it must never receive exception text.
        code = 121
    raise SystemExit(code)
