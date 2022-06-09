import json

import requests
from requests.structures import CaseInsensitiveDict

ip = "192.168.178.63"

def get_token():
    login = {
        "password": "boschrexroth",
        "name": "boschrexroth"
    }
    url = f"https://{ip}/identity-manager/api/v2/auth/token"
    res = requests.post(url=url, json=login, verify=False)
    return json.loads(res.text)["access_token"]


def motion_trigger(bearer_token=None):
    if not bearer_token:
        bearer_token = get_token()
    url = f"https://{ip}/automation/api/v2/nodes/plc/app/Application/sym/PLC_PRG/trigger"

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    headers["Authorization"] = f"Bearer {bearer_token}"
    data = {
        "schema": "types/datalayer/metadata",
        "type": "bool8",
        "value": True
    }
    res = requests.put(url=url, headers=headers, json=data, verify=False)
    if res.text.strip() == '{"type":"bool8","value":true}':
        return True
    return False


if __name__ == "__main__":
    assert motion_trigger()