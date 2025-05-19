from .copy_checkpoints import get_checkpoint
import os
from typing import Optional
from .run_command import run_command
import shutil
import json
from .dirs import get_dirs


def validate(run, patch_ckpt=None, bs: Optional[int] = None, flags: str = ""):
    get_checkpoint(run)

    ckpt_dir = f"checkpoints/{run.id}/"
    res_path = f"validation_results/{run.id}.json"

    curr_dir, main_dir, my_rel_dir = get_dirs()

    signature = {
        "flags": flags,
        "bs": bs
    }

    if not os.path.isfile(res_path):
        ckpt_path = f"{ckpt_dir}/model.pth"
        os.makedirs(os.path.dirname(res_path), exist_ok=True)

        os.chdir(main_dir)

        ckpt_path = f"{my_rel_dir}/{ckpt_path}"
        if patch_ckpt is not None:
            ckpt_path = patch_ckpt(ckpt_path)

        if bs is None:
            bs = ""
        else:
            bs = f"--batch_size {bs}"

        shutil.rmtree("save/post_validate", ignore_errors=True)

        cmd = f"python3 main.py --name post_validate --log tb --restore {ckpt_path} --test_only 1 -reset 1 --keep_alive 0 {bs} {flags}"
        print("Validate command: ", cmd)
        out = run_command(cmd)
        lines = out.splitlines()
        start_line = lines.index('Validate returned:')
        end_line = None
        for i in range(start_line, len(lines)):
            if lines[i].startswith("-------"):
                end_line = i
                break

        assert end_line is not None

        res = "\n".join(lines[start_line+1:end_line])
        os.chdir(curr_dir)

        res = json.loads(res)
        res["__signature__"] = signature

        with open(res_path, "w") as f:
            json.dump(res, f)

    with open(res_path, "r") as f:
        res = json.load(f)

    if signature != res.get("__signature__", {}):
        os.remove(res_path)
        return validate(run, patch_ckpt, bs, flags)

    return res