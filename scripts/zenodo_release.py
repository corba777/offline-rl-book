#!/usr/bin/env python3
"""
Publish a release as a new version of the book's Zenodo record, with files attached.

Zenodo's GitHub integration archives only the source zip; this script uploads
the release PDFs as well, through the Zenodo deposit API. It creates a new
version of the concept record (so the all-versions DOI stays the same),
replaces the files carried over from the previous version, sets the metadata
from .zenodo.json with the version and date of this release, and publishes.

Usage:
    ZENODO_TOKEN=... python scripts/zenodo_release.py --version 1.1.4 dist/*.pdf dist/*.zip
    --draft     stop before publishing, to inspect the draft on zenodo.org
ZENODO_URL selects another instance, e.g. https://sandbox.zenodo.org with a sandbox
record id in --concept. The token needs the deposit:write and deposit:actions scopes.
Standard library only, so the workflow needs no installs.
"""
import argparse
import datetime
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONCEPT = "24045871"          # concept record: 10.5281/zenodo.24045871 resolves to the latest version


def api(method, url, token, body=None, data=None, content_type="application/json"):
    if body is not None:
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": content_type,
                                          "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url} -> {e.code}\n{e.read().decode(errors='replace')}")
    return json.loads(raw) if raw else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--version", required=True, help="e.g. 1.1.4 (no leading v)")
    p.add_argument("--concept", default=CONCEPT)
    p.add_argument("--draft", action="store_true", help="do not publish")
    args = p.parse_args()
    token = os.environ.get("ZENODO_TOKEN") or sys.exit("ZENODO_TOKEN is not set")
    base = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/")
    for f in args.files:
        if not f.is_file():
            sys.exit(f"missing file: {f}")

    latest = api("GET", f"{base}/api/records/{args.concept}/versions/latest", token)
    print(f"latest version: {latest['id']} ({latest.get('metadata', {}).get('version')})")
    if latest.get("metadata", {}).get("version") == args.version:
        sys.exit(f"version {args.version} is already published on Zenodo")
    nv = api("POST", f"{base}/api/deposit/depositions/{latest['id']}/actions/newversion", token)
    draft = api("GET", nv["links"]["latest_draft"], token)
    dep = f"{base}/api/deposit/depositions/{draft['id']}"
    print(f"draft: {draft['id']}")

    # A new version starts with the previous version's files; replace them.
    for f in draft.get("files", []):
        api("DELETE", f"{dep}/files/{f['id']}", token)
    for f in args.files:
        print(f"uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        api("PUT", f"{draft['links']['bucket']}/{f.name}", token, data=f.read_bytes(),
            content_type="application/octet-stream")

    meta = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    meta.update(version=args.version, publication_date=datetime.date.today().isoformat())
    api("PUT", dep, token, body={"metadata": meta})

    if args.draft:
        print(f"draft left unpublished: {draft['links'].get('html', dep)}")
        return
    pub = api("POST", f"{dep}/actions/publish", token)
    print(f"published: https://doi.org/{pub['doi']}  ({pub['links'].get('html', '')})")


if __name__ == "__main__":
    main()
