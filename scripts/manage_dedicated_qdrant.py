"""Create/start/check the dedicated DocuzRAG Qdrant Docker instance.

This uses the Docker Engine API because some development environments expose
the host daemon without installing the Docker CLI in the project shell. It
only touches the explicitly named DocuzRAG container and volume.
"""

from __future__ import annotations

import argparse
import time

import httpx


ENGINE = "http://host.docker.internal:2375"
CONTAINER = "docuzrag-qdrant"
VOLUME = "docuzrag_qdrant_data"
IMAGE = "qdrant/qdrant:v1.12.5"
PUBLIC_HTTP = 6335
PUBLIC_GRPC = 6336


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    client = httpx.Client(base_url=ENGINE, timeout=30)
    version = client.get("/version")
    version.raise_for_status()
    if not args.check_only:
        _ensure_volume(client)
        _ensure_container(client)
    _wait_for_qdrant()
    print(f"{CONTAINER} is healthy on host port {PUBLIC_HTTP}; volume {VOLUME} is in use.")


def _ensure_volume(client: httpx.Client) -> None:
    response = client.get(f"/volumes/{VOLUME}")
    if response.status_code == 200:
        return
    if response.status_code != 404:
        response.raise_for_status()
    created = client.post(
        "/volumes/create",
        json={"Name": VOLUME, "Driver": "local", "Labels": {"com.docuzrag.role": "qdrant"}},
    )
    created.raise_for_status()


def _ensure_container(client: httpx.Client) -> None:
    response = client.get(f"/containers/{CONTAINER}/json")
    if response.status_code == 404:
        created = client.post(
            "/containers/create",
            params={"name": CONTAINER},
            json={
                "Image": IMAGE,
                "Env": ["QDRANT__LOG_LEVEL=INFO"],
                "ExposedPorts": {"6333/tcp": {}, "6334/tcp": {}},
                "HostConfig": {
                    "Binds": [f"{VOLUME}:/qdrant/storage"],
                    "PortBindings": {
                        "6333/tcp": [{"HostPort": str(PUBLIC_HTTP)}],
                        "6334/tcp": [{"HostPort": str(PUBLIC_GRPC)}],
                    },
                    "RestartPolicy": {"Name": "unless-stopped"},
                },
                "Healthcheck": {
                    "Test": ["CMD", "/qdrant/qdrant", "--version"],
                    "Interval": 10_000_000_000,
                    "Timeout": 5_000_000_000,
                    "Retries": 5,
                },
                "Labels": {"com.docuzrag.service": "qdrant"},
            },
        )
        created.raise_for_status()
        response = client.get(f"/containers/{CONTAINER}/json")
    response.raise_for_status()
    details = response.json()
    if details["Config"]["Image"] != IMAGE:
        raise RuntimeError(f"{CONTAINER} already exists with a different image.")
    if details["HostConfig"]["PortBindings"]["6333/tcp"][0]["HostPort"] != str(PUBLIC_HTTP):
        raise RuntimeError(f"{CONTAINER} does not use the dedicated host port {PUBLIC_HTTP}.")
    if details["State"]["Status"] != "running":
        started = client.post(f"/containers/{CONTAINER}/start")
        if started.status_code not in {204, 304}:
            started.raise_for_status()


def _wait_for_qdrant() -> None:
    for _ in range(30):
        try:
            response = httpx.get(f"http://host.docker.internal:{PUBLIC_HTTP}/healthz", timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError(f"Dedicated Qdrant did not become healthy on port {PUBLIC_HTTP}.")


if __name__ == "__main__":
    main()
