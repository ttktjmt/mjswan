"""CLI entry points for mjswan scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="mjswan",
    help="Browser-based MuJoCo simulation with real-time policy control.",
    no_args_is_help=True,
)
console = Console()


def _run_module(module_path: str) -> None:
    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-m", module_path],
        check=False,
        cwd=project_root,
    )
    sys.exit(result.returncode)


def _fmt_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


# ── view ──────────────────────────────────────────────────────


@app.command("view")
def view_cmd(
    model: Annotated[Path, typer.Argument(help="Path to MuJoCo XML/MJCF file.")],
    name: Annotated[
        str, typer.Option(help="Scene name shown in the viewer.")
    ] = "Scene",
    port: Annotated[int, typer.Option(help="HTTP server port.")] = 8080,
    host: Annotated[str, typer.Option(help="HTTP server host.")] = "localhost",
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open browser automatically.")
    ] = False,
) -> None:
    """View a MuJoCo XML/MJCF file in the browser."""
    import tempfile

    import mujoco

    from mjswan import Builder

    if not model.exists():
        console.print(f"[red]Error:[/red] File not found: {model}")
        raise typer.Exit(1)

    spec = mujoco.MjSpec.from_file(str(model.resolve()))
    builder = Builder()
    builder.add_project(name=model.stem).add_scene(spec=spec, name=name)

    with tempfile.TemporaryDirectory() as tmp:
        built_app = builder.build(output_dir=tmp)
        built_app.launch(host=host, port=port, open_browser=not no_open)


# ── serve ─────────────────────────────────────────────────────


@app.command("serve")
def serve_cmd(
    source: Annotated[
        Path,
        typer.Argument(help="A built mjswan dist directory, or a .swn document."),
    ],
    port: Annotated[int, typer.Option(help="HTTP server port.")] = 8080,
    host: Annotated[str, typer.Option(help="HTTP server host.")] = "localhost",
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Do not open browser automatically.")
    ] = False,
    height: Annotated[int, typer.Option(help="Colab iframe height in pixels.")] = 600,
) -> None:
    """Serve a pre-built mjswan app, from a dist directory or a .swn document."""
    from mjswan.app import MjswanApp
    from mjswan.document import DocumentError, is_document

    if not source.exists():
        console.print(f"[red]Error:[/red] Not found: {source}")
        raise typer.Exit(1)
    if is_document(source):
        # Unpacking a large document takes a few seconds; say what the pause is.
        console.print(f"Expanding {source.name} with the packaged engine...")
    try:
        app_to_serve = MjswanApp.from_document(source)
    except DocumentError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    app_to_serve.launch(host=host, port=port, open_browser=not no_open, height=height)


# ── publish ───────────────────────────────────────────────────


@app.command("publish")
def publish_cmd(
    dist_dir: Annotated[
        Path,
        typer.Argument(help="A built mjswan dist directory, or a .swn document."),
    ],
    *,
    title: Annotated[
        Optional[str],
        typer.Option(help="Simulation title. Defaults to the first project's name."),
    ] = None,
    description: Annotated[
        Optional[str], typer.Option(help="Optional description.")
    ] = None,
    tag: Annotated[
        Optional[list[str]],
        typer.Option(help="Tag to attach (repeatable)."),
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(help="Supabase access token. Falls back to $MJSWAN_TOKEN."),
    ] = None,
    api_base: Annotated[
        Optional[str],
        typer.Option(
            help="Cloud API base URL. Falls back to $MJSWAN_API_BASE, then "
            "https://api.mjswan.com."
        ),
    ] = None,
) -> None:
    """Publish a built dist directory, or a .swn document, to mjswan Cloud."""
    from mjswan.publish import (
        TOKEN_ENV_VAR,
        PublishError,
        publish_dist,
        simulation_url,
    )

    resolved = dist_dir.expanduser().resolve()
    if not resolved.exists():
        console.print(f"[red]Error:[/red] Not found: {dist_dir}")
        raise typer.Exit(1)

    # No flag, env or stored session: run the browser login first, then publish.
    import os

    from mjswan import auth

    if not token and not os.environ.get(TOKEN_ENV_VAR) and not auth.load_credentials():
        console.print("[dim]Not logged in — signing in to mjswan Cloud first…[/dim]")
        if not _do_login(open_browser=True):
            raise typer.Exit(1)

    try:
        result = publish_dist(
            resolved,
            title=title,
            description=description,
            tags=list(tag) if tag else None,
            token=token,
            api_base=api_base,
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
            on_warning=lambda msg: console.print(f"[yellow]Warning:[/yellow] {msg}"),
        )
    except PublishError as exc:
        location = f" [dim]({exc.file})[/dim]" if exc.file else ""
        console.print(f"[red]Publish failed:[/red] {exc}{location}")
        raise typer.Exit(1)

    url = simulation_url(result.id)
    console.print(f"[green]Published![/green] [bold][link={url}]{url}[/link][/bold]")


# ── login / logout / whoami ────────────────────────────────────


def _do_login(*, open_browser: bool) -> bool:
    """Run the OAuth flow and report which account signed in.

    Returns ``True`` on success, ``False`` (with an error printed) on failure.
    Shared by ``mjswan login`` and ``mjswan publish``'s auto-login.
    """
    from mjswan.auth import AuthError, login

    try:
        creds = login(
            open_browser=open_browser,
            on_progress=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
    except AuthError as exc:
        console.print(f"[red]Login failed:[/red] {exc}")
        return False

    who = f" as [bold]{creds.username}[/bold]" if creds.username else ""
    console.print(f"[green]Logged in to mjswan Cloud{who}.[/green]")
    return True


@app.command("login")
def login_cmd(
    no_open: Annotated[
        bool,
        typer.Option(
            "--no-open", help="Do not open the browser; print the URL instead."
        ),
    ] = False,
) -> None:
    """Sign in to mjswan Cloud via GitHub (loopback OAuth)."""
    if not _do_login(open_browser=not no_open):
        raise typer.Exit(1)


@app.command("whoami")
def whoami_cmd() -> None:
    """Show the mjswan Cloud account you are signed in as."""
    from mjswan.auth import AuthError, fetch_identity

    try:
        identity = fetch_identity()
    except AuthError as exc:
        # Session exists locally but is no longer valid server-side.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if identity is None:
        console.print("[dim]Not logged in. Run [bold]mjswan login[/bold].[/dim]")
        raise typer.Exit(1)

    name = identity.username or identity.user_id
    detail = f" [dim]({identity.email})[/dim]" if identity.email else ""
    console.print(f"Logged in as [bold]{name}[/bold]{detail}")


@app.command("logout")
def logout_cmd() -> None:
    """Remove the stored mjswan Cloud session."""
    from mjswan.auth import clear_credentials, credentials_path

    if clear_credentials():
        console.print("[green]Logged out.[/green]")
    else:
        console.print(
            f"[dim]Not logged in (no credentials at {credentials_path()}).[/dim]"
        )


# ── new ───────────────────────────────────────────────────────

_TEMPLATES: dict[str, dict[str, str]] = {
    "hello-world": {
        "main.py": """\
import mujoco

import mjswan


def main() -> None:
    builder = mjswan.Builder()
    project = builder.add_project(name="{name}")

    spec = mujoco.MjSpec.from_file("model.xml")
    project.add_scene(spec=spec, name="Scene")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
""",
        "model.xml": """\
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
    },
    "policy": {
        "main.py": """\
import mujoco
import onnx

import mjswan


def main() -> None:
    builder = mjswan.Builder()
    project = builder.add_project(name="{name}")

    spec = mujoco.MjSpec.from_file("model.xml")
    scene = project.add_scene(spec=spec, name="Scene")

    # Replace with your ONNX policy file
    policy_model = onnx.load("policy.onnx")
    scene.add_policy(policy=policy_model, name="Policy")

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
""",
        "model.xml": """\
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""",
    },
    "mjlab": {
        "main.py": """\
import mjswan


def main() -> None:
    # Replace "go2_flat" with your mjlab task ID
    app = mjswan.Builder.from_mjlab("go2_flat").build()
    app.launch()


if __name__ == "__main__":
    main()
""",
    },
}


@app.command("new")
def new_cmd(
    name: Annotated[
        str, typer.Argument(help="Project name (also used as directory name).")
    ],
    template: Annotated[
        str, typer.Option(help="Template to use: hello-world | policy | mjlab.")
    ] = "hello-world",
) -> None:
    """Scaffold a new mjswan project from a template."""
    if template not in _TEMPLATES:
        console.print(
            f"[red]Error:[/red] Unknown template '{template}'. "
            f"Available: {', '.join(_TEMPLATES)}"
        )
        raise typer.Exit(1)

    project_dir = Path(name)
    if project_dir.exists():
        console.print(f"[red]Error:[/red] Directory '{name}' already exists.")
        raise typer.Exit(1)

    project_dir.mkdir()
    for filename, content in _TEMPLATES[template].items():
        (project_dir / filename).write_text(content.format(name=name))
        console.print(f"  [green]created[/green]  {name}/{filename}")

    console.print(f"\n[bold]Done![/bold] Start with:\n  cd {name}\n  python main.py")


# ── demo ──────────────────────────────────────────────────────

_DEMOS: dict[str, str] = {
    "simple": "examples.demo.simple",
    "main": "examples.demo.main",
    "mjlab": "examples.mjlab.defaults.main",
}


@app.command("demo")
def demo_cmd(
    name: Annotated[
        Optional[str], typer.Argument(help="Demo name. Omit to run 'simple'.")
    ] = None,
    list_: Annotated[
        bool, typer.Option("--list", "-l", help="List available demos.")
    ] = False,
) -> None:
    """Run a built-in mjswan demo."""
    if list_:
        console.print("[bold]Available demos:[/bold]")
        for demo_name in _DEMOS:
            console.print(f"  {demo_name}")
        return

    demo_name = name or "simple"
    if demo_name not in _DEMOS:
        console.print(
            f"[red]Error:[/red] Unknown demo '{demo_name}'. "
            "Run [bold]mjswan demo --list[/bold] to see available demos."
        )
        raise typer.Exit(1)

    _run_module(_DEMOS[demo_name])


# ── info ──────────────────────────────────────────────────────


@app.command("info")
def info_cmd(
    dist_dir: Annotated[
        Path,
        typer.Argument(help="A built mjswan dist directory, or a .swn document."),
    ],
) -> None:
    """Show information about a built mjswan app or a .swn simulation document."""
    from rich.tree import Tree

    from mjswan.document import (
        MANIFEST_NAME,
        DocumentError,
        as_directory,
        is_document,
        read_manifest,
    )

    if not dist_dir.exists():
        console.print(f"[red]Error:[/red] Not found: {dist_dir}")
        raise typer.Exit(1)
    if not is_document(dist_dir) and not (dist_dir / MANIFEST_NAME).exists():
        console.print(f"[red]Error:[/red] No {MANIFEST_NAME} found in {dist_dir}")
        raise typer.Exit(1)

    try:
        with as_directory(dist_dir) as root:
            manifest = read_manifest(root)
            kind = "document" if is_document(dist_dir) else "app"
            version = manifest.get("version", "unknown")
            fmt = manifest.get("format", "?")
            tree = Tree(
                f"[bold]mjswan {kind}[/bold] — {dist_dir}  "
                f"[dim]v{version}, format {fmt}[/dim]"
            )
            total_bytes = _describe_projects(tree, root, manifest)
    except DocumentError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    tree.add(f"[dim]Total scene+policy assets: {_fmt_size(total_bytes)}[/dim]")
    console.print(tree)


def _describe_licenses(node, root: Path, directory: Path) -> None:
    """One line per license file in ``directory`` (ADR 0007 §4), identifier included."""
    from mjswan.licenses import declare

    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if not path.is_file():
            continue
        declaration = declare(path.relative_to(root).as_posix(), path.read_bytes())
        if declaration is None:
            continue
        label = "License" if declaration.location.kind == "LICENSE" else "Notice"
        what = f": [blue]{declaration.spdx}[/blue]" if declaration.spdx else ""
        node.add(f"{label}{what}  [dim]{declaration.path}[/dim]")


def _describe_projects(tree, root: Path, manifest: dict) -> int:
    """Add one node per project/scene/MDP/policy under ``tree``; return the asset bytes."""
    total_bytes = 0
    for project in manifest.get("projects", []):
        p_node = tree.add(
            f"[cyan]{project['name']}[/cyan]  [dim][{project['id']}][/dim]"
        )
        _describe_licenses(p_node, root, root / project["id"])
        for scene in project.get("scenes", []):
            scene_dir = root / project["id"] / scene["id"]
            scene_path = scene_dir / scene.get("scene", "")
            scene_size = scene_path.stat().st_size if scene_path.exists() else 0
            total_bytes += scene_size
            s_node = p_node.add(
                f"[green]{scene['name']}[/green]  "
                f"[dim]{scene['id']}/{scene.get('scene', '')}  "
                f"({_fmt_size(scene_size)})[/dim]"
            )
            _describe_licenses(s_node, root, scene_dir)
            for mdp in scene.get("mdps", []):
                graphs = list((scene_dir / "mdp" / mdp["id"]).rglob("*.onnx"))
                s_node.add(
                    f"MDP: [magenta]{mdp['id']}[/magenta]  [dim]{len(graphs)} graph(s)[/dim]"
                )
            for policy in scene.get("policies", []):
                onnx_path = scene_dir / policy.get("onnx", "")
                policy_size = onnx_path.stat().st_size if onnx_path.exists() else 0
                total_bytes += policy_size
                size_str = f"  ({_fmt_size(policy_size)})" if policy_size else ""
                s_node.add(
                    f"Policy: [yellow]{policy['name']}[/yellow]  "
                    f"[dim]mdp={policy.get('mdp')}{size_str}[/dim]"
                )
    return total_bytes


# ── Legacy entry points (backward compatibility) ──────────────


def main() -> None:
    """Run examples/demo/main.py"""
    _run_module("examples.demo.main")


def simple() -> None:
    """Run examples/demo/simple.py"""
    _run_module("examples.demo.simple")


def mjlab() -> None:
    """Run examples/mjlab/defaults/main.py"""
    _run_module("examples.mjlab.defaults.main")


def serve() -> None:
    """Launch a pre-built mjswan app from a dist directory or a ``.swn`` document.

    Usage: serve <dist-dir | document.swn>
    """
    if len(sys.argv) < 2:
        print("Usage: serve <dist-dir | document.swn>", file=sys.stderr)
        sys.exit(1)

    from mjswan.app import MjswanApp

    MjswanApp.from_document(Path(sys.argv[1])).launch()
