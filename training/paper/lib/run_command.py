import subprocess
from typing import Optional


def run_command(cmd: str, get_stderr: bool = False, allow_failure: bool = False, find_beginend: bool = False) -> Optional[str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE if get_stderr else None,
                            shell=True, stdin=subprocess.PIPE)
    # input.encode() if input is not None else None
    res = proc.communicate(None)
    stdout = res[0].decode()
    if proc.returncode != 0:
        if allow_failure:
            return None
        stderr = res[1].decode()
        raise RuntimeError(f"Command {cmd} failed with return code {proc.returncode} and stderr: {stderr}")

    if find_beginend:
        needed = []
        adding = False
        for line in stdout.splitlines():
            if line == '------------------- START':
                adding = True
            elif line == '------------------- END':
                adding = False
            elif adding:
                needed.append(line)

        stdout = "\n".join(needed)
    return stdout