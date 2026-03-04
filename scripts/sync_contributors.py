"""Fetch GitHub contributors and update src/mjswan/template/src/Version.ts.

Usage:
    uv run sync_contributors.py
"""

import json
import pathlib
import re
import urllib.request


def main() -> None:
    version_path = (
        pathlib.Path(__file__).parent.parent / "src/mjswan/template/src/Version.ts"
    )

    api_url = "https://api.github.com/repos/ttktjmt/mjswan/contributors?per_page=500"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": "mjswan-sync-script",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    with urllib.request.urlopen(req) as response:
        contributors_json = json.loads(response.read().decode())

    contributors_data = [
        {"login": c["login"], "html_url": c["html_url"]}
        for c in contributors_json
        if isinstance(c, dict) and c.get("type") == "User"
    ]
    print(f"Fetched {len(contributors_data)} contributors from GitHub")

    entries = "\n".join(
        f'  {{\n    login: "{c["login"]}",\n    html_url: "{c["html_url"]}",\n  }},'
        for c in contributors_data
    )
    contributors_block = (
        "// GitHub contributors for the mjswan project.\n"
        "// Run `uv run sync_contributors.py` to update.\n"
        "export interface Contributor {\n"
        "  login: string;\n"
        "  html_url: string;\n"
        "}\n\n"
        f"export const GITHUB_CONTRIBUTORS: Contributor[] = [\n{entries}\n];"
    )

    content = version_path.read_text()
    # Replace everything from the contributors block marker to end of file
    content = re.sub(
        r"// GitHub contributors for the mjswan project\..*",
        contributors_block,
        content,
        flags=re.DOTALL,
    )
    version_path.write_text(content)
    print(f"Updated {version_path}")


if __name__ == "__main__":
    main()
