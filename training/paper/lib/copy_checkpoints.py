import os
import getpass
import paramiko
import scp
from .get_logs import get_logs
import torch
import sys
from .config import get_config

scp_password = None

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)+"/../../.."))


def proxyjump():
    global scp_password

    config = get_config()
    jump_host = config.get("proxyjump", None)
    jump_port = 22
    jump_user = config["username"]
    jump_password = scp_password

    target_host = config["host"]
    target_port = 22
    target_user = config["username"]
    target_password = scp_password

    if jump_host is not None:
        # Establish a connection to the jump host
        jump_client = paramiko.SSHClient()
        jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        jump_client.connect(jump_host, username=jump_user, password=jump_password, port=jump_port)

        transport = jump_client.get_transport()

        dest_addr = (target_host, 22)
        local_addr = ('127.0.0.1', 22)
        channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
    else:
        channel = None

    target_client = paramiko.SSHClient()
    target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # Connect to the target host through the jump host
    target_client.connect(target_host, username=target_user, password=target_password, port=target_port, sock=channel)

    return target_client, jump_client


def find_checkpoint(run):
    logs = get_logs(run)
    logs = list(reversed(logs))

    prefix = "Saving "
    for l in logs:
        if l.startswith(prefix) and l.endswith(".pth"):
            path = l[len(prefix):]
            break
    else:
        raise RuntimeError(f"Path not found for run {run.id}")

    progname = run.metadata["program"]
    basedir = os.path.dirname(progname)

    return os.path.join(basedir, path)


def scp_file_run(ckpt, target):
    client, jump_client = proxyjump()

    # Open an SFTP session on the jump host
    with scp.SCPClient(client.get_transport()) as sc:
        sc.get(ckpt, target)

    # Close the connections
    client.close()
    if jump_client is not None:
        jump_client.close()


def get_checkpoint(run, strip=True):
    config = get_config()

    global scp_password
    run_id = run.id
    ckpt_dir = f"checkpoints/{run_id}"
    model_file = f"{ckpt_dir}/model.pth"
    model_file_downloaded = model_file + (".tmp" if strip else "")

    if os.path.isfile(model_file):
        return model_file

    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Downloading checkpoint for {run.id}...")
    hostname = run.metadata["host"]

    if not hostname.endswith(config["host_suffix"]):
        hostname = hostname + config["host_suffix"]

    ckpt = find_checkpoint(run)

    if scp_password is None:
        scp_password = getpass.getpass("Scp password: ")

    scp_file_run(ckpt, model_file_downloaded)

    assert os.path.isfile(model_file_downloaded), "Failed to download checkpoint"

    if strip:
        print("Stripping checkpoint...")
        ckpt = torch.load(model_file_downloaded, weights_only=False)
        del ckpt["optimizer"]
        torch.save(ckpt, model_file)
        os.remove(model_file_downloaded)

    print(f"Downloaded checkpoint for {run.id} to {model_file}")
    return model_file
