"""CLI entry points for mjswan scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape

from mjswan import __version__

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
    from mjswan.cloud.publish import (
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

    from mjswan.cloud import auth

    if not token and not os.environ.get(TOKEN_ENV_VAR) and not auth.load_credentials():
        console.print("[dim]Not logged in: signing in to mjswan Cloud first…[/dim]")
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
            attest=_confirm_avatar_use,
        )
    except PublishError as exc:
        location = f" [dim]({exc.file})[/dim]" if exc.file else ""
        console.print(f"[red]Publish failed:[/red] {exc}{location}")
        raise typer.Exit(1)

    url = simulation_url(result.id)
    console.print(f"[green]Published![/green] [bold][link={url}]{url}[/link][/bold]")


def _confirm_avatar_use(path: str, terms) -> bool:
    """Ask whether the publisher may use *path* as an avatar; without a terminal, no."""
    from mjswan.license.vrm import describe

    who = "its author" if terms.avatar_use == "author" else "people its author licensed"
    console.print(
        f"[bold]{escape(path)}[/bold]: {escape(describe(terms))}. Its license lets only "
        f"{who} use it as an avatar, and mjswan Cloud will show that you said you may."
    )
    if not _interactive():
        return False
    return typer.confirm("Are you its author, or licensed by them?", default=False)


def _interactive() -> bool:
    return sys.stdin.isatty()


# ── login / logout / whoami ────────────────────────────────────


def _do_login(*, open_browser: bool) -> bool:
    """Run the OAuth flow and report which account signed in.

    Returns ``True`` on success, ``False`` (with an error printed) on failure.
    Shared by ``mjswan login`` and ``mjswan publish``'s auto-login.
    """
    from mjswan.cloud.auth import AuthError, login

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
    from mjswan.cloud.auth import AuthError, fetch_identity

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
    from mjswan.cloud.auth import clear_credentials, credentials_path

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
    # Your policy's control period in seconds (mjlab: timestep * decimation).
    scene = project.add_scene(spec=spec, name="Scene", control_dt=0.02)

    # Replace with your ONNX policy; add_policy needs its observations= and actions=.
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

#: Name → (module it runs, what it builds). The modules live under `examples/`, which a
#: source checkout has and an installed wheel does not; `tests/test_cli_demo.py` checks
#: each one exists.
_DEMOS: dict[str, tuple[str, str]] = {
    "main": ("examples.demo.main", "Eight mjlab tasks and a showcase, as deployed"),
    "simple": ("examples.demo.simple", "One mjlab task, one checkpoint"),
    "mujoco": ("examples.demo.mujoco_models", "MuJoCo's replicate gallery, no policy"),
}


@app.command("demo")
def demo_cmd(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Demo to run. Omit to list what there is."),
    ] = None,
) -> None:
    """Run a built-in mjswan demo, or list them."""
    # No name lists instead of running one: every demo downloads models and checkpoints.
    if name is None:
        console.print("[bold]Available demos:[/bold]")
        width = max(len(demo) for demo in _DEMOS)
        for demo, (_, description) in _DEMOS.items():
            console.print(f"  [bold]{demo:<{width}}[/bold]  {description}")
        console.print("\nRun one with [bold]mjswan demo <name>[/bold].")
        return

    if name not in _DEMOS:
        console.print(
            f"[red]Error:[/red] Unknown demo '{name}'. "
            "Run [bold]mjswan demo[/bold] to see what there is."
        )
        raise typer.Exit(1)

    _run_module(_DEMOS[name][0])


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
                f"[bold]mjswan {kind}[/bold]: {escape(str(dist_dir))}  "
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
    from mjswan.license import declare

    for path in sorted(directory.iterdir()) if directory.is_dir() else []:
        if not path.is_file():
            continue
        declaration = declare(path.relative_to(root).as_posix(), path.read_bytes())
        if declaration is None:
            continue
        label = "License" if declaration.location.kind == "LICENSE" else "Notice"
        what = f": [blue]{escape(declaration.spdx)}[/blue]" if declaration.spdx else ""
        node.add(f"{label}{what}  [dim]{escape(declaration.path)}[/dim]")


def _describe_skin(node, scene_dir: Path, skin: dict) -> None:
    """One line for a skin, with a VRM's license."""
    from mjswan.document import read_skin_file
    from mjswan.license.vrm import describe, vrm_license

    path = scene_dir / skin.get("path", "")
    size = path.stat().st_size if path.is_file() else 0
    detail = ""
    if skin.get("format") == "vrm" and size:
        try:
            file = read_skin_file(path)
        except ValueError as exc:
            detail = f"  [red]{escape(str(exc))}[/red]"
        else:
            if file.vrm_version is not None:
                terms = vrm_license(file.meta, file.vrm_version)
                detail = f"  [blue]{escape(describe(terms))}[/blue]"
    node.add(
        f"Skin: [yellow]{escape(skin.get('name', ''))}[/yellow]  "
        f"[dim]{escape(skin.get('path', ''))} ({_fmt_size(size)})[/dim]{detail}"
    )


def _describe_projects(tree, root: Path, manifest: dict) -> int:
    """Add one node per project/scene/MDP/policy under ``tree``; return the asset bytes.

    Names, ids and paths are escaped: rich would parse a ``[...]`` in them as markup.
    """
    total_bytes = 0
    for project in manifest.get("projects", []):
        project_id = escape(f"[{project['id']}]")
        p_node = tree.add(
            f"[cyan]{escape(project['name'])}[/cyan]  [dim]{project_id}[/dim]"
        )
        _describe_licenses(p_node, root, root / project["id"])
        for scene in project.get("scenes", []):
            scene_dir = root / project["id"] / scene["id"]
            scene_path = scene_dir / scene.get("scene", "")
            scene_size = scene_path.stat().st_size if scene_path.exists() else 0
            total_bytes += scene_size
            s_node = p_node.add(
                f"[green]{escape(scene['name'])}[/green]  "
                f"[dim]{escape(scene['id'] + '/' + scene.get('scene', ''))}  "
                f"({_fmt_size(scene_size)})[/dim]"
            )
            _describe_licenses(s_node, root, scene_dir)
            for skin in scene.get("skins", []):
                _describe_skin(s_node, scene_dir, skin)
            for mdp in scene.get("mdps", []):
                graphs = list((scene_dir / "mdp" / mdp["id"]).rglob("*.onnx"))
                s_node.add(
                    f"MDP: [magenta]{escape(mdp['id'])}[/magenta]  "
                    f"[dim]{len(graphs)} graph(s)[/dim]"
                )
            for policy in scene.get("policies", []):
                onnx_path = scene_dir / policy.get("onnx", "")
                policy_size = onnx_path.stat().st_size if onnx_path.exists() else 0
                total_bytes += policy_size
                size_str = f"  ({_fmt_size(policy_size)})" if policy_size else ""
                s_node.add(
                    f"Policy: [yellow]{escape(policy['name'])}[/yellow]  "
                    f"[dim]mdp={escape(str(policy.get('mdp')))}{size_str}[/dim]"
                )
    return total_bytes


# ── skin ──────────────────────────────────────────────────────

skin_app = typer.Typer(
    name="skin",
    help="Skins: rigged characters drawn over a robot.",
    no_args_is_help=True,
)
app.add_typer(skin_app)


@skin_app.command("map")
def skin_map_cmd(
    file: Annotated[Path, typer.Argument(help="The skin: a .vrm or a .glb.")],
    *,
    robot: Annotated[
        Optional[str],
        typer.Option(help="The robot: an MJCF .xml, a .mjb, or an mjlab task id."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write a SkinConfig draft here."),
    ] = None,
) -> None:
    """Draft a skin's maps: which node is which bone, and which robot body each follows.

    The guesses are for you to check, line by line, before you use them.
    """
    from rich.table import Table

    from mjswan.document import read_skin_file
    from mjswan.rig import guess as g

    try:
        skin = read_skin_file(file.expanduser())
    except (OSError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1)
    model = _load_robot(robot) if robot else None

    guess = None if skin.format == "vrm" else g.guess_humanoid(skin)
    nodes = g.in_vrm_order(skin.human_bones if guess is None else guess.humanoid)
    bones = g.guess_bones(model, nodes) if model is not None else {}
    shown = {b: n for b, n in nodes.items() if b in bones} if bones else nodes
    kind = "VRM " + (skin.vrm_version or "") if guess is None else f"glTF, {guess.kind}"
    facing = "" if guess is None else f", facing {guess.facing}"
    size = f"{len(skin.joints)} joints" if skin.joints else f"{len(skin.names)} nodes"
    console.print(f"[bold]{escape(skin.path.name)}[/bold]: {kind}, {size}{facing}")

    scale = None
    if guess is not None and model is not None:
        found = g.guess_scale(skin, guess, model, bones)
        if found:
            scale = found[0]
            console.print(f"  scale={scale:.2g}   [dim]{escape(found[1])}[/dim]")

    table = Table(box=None, pad_edge=False)
    for column in ("bone", "node", "body", "how"):
        table.add_column(column, style="dim" if column == "how" else None)
    for bone, node in shown.items():
        how = guess.why.get(bone, "") if guess is not None else ""
        table.add_row(bone, escape(skin.names[node]), escape(bones.get(bone, "")), how)
    if shown:
        console.print(table)
    if len(shown) < len(nodes):
        console.print(
            f"  [dim]{len(nodes) - len(shown)} more bones, no body to follow[/dim]"
        )
    for n in guess.springs if guess is not None else []:
        console.print(
            f"  springs?  {escape(skin.names[n])}  [dim]a chain no bone takes[/dim]"
        )
    for limb in guess.limbs if guess is not None else []:
        named = ", ".join(skin.names[n] for n in limb)
        near = ", ".join(g.close_bodies(model, skin.names[limb[0]])) if model else ""
        console.print(f"  limb?  {escape(named)}  [dim]{escape(near)}[/dim]")

    rest_pose = _rest_pose(model, guess, bones) if model is not None else None
    if output:
        draft = g.draft_source(
            " ".join(
                ["mjswan skin map", file.name, *(["--robot", robot] if robot else [])]
            ),
            source=str(file),
            guess=guess,
            names=skin.names,
            bones=bones or {bone: "" for bone in nodes},
            scale=scale,
            rest_pose=rest_pose,
        )
        output.write_text(draft)
        console.print(
            f"Wrote {escape(str(output))}: a SkinConfig draft to check and paste."
        )


def _load_robot(robot: str):
    """The robot's compiled model, from a file or an mjlab task."""
    import mujoco

    path = Path(robot).expanduser()
    try:
        if path.suffix == ".mjb":
            return mujoco.MjModel.from_binary_path(str(path))
        if path.suffix == ".xml":
            return mujoco.MjModel.from_xml_path(str(path))
        from mjlab.scene import Scene
        from mjlab.tasks.registry import load_env_cfg

        from mjswan.mjlab import ensure_mjlab_extensions
    except ImportError:
        console.print(
            f"[red]Error:[/red] {escape(robot)} is not a .xml or .mjb, and an mjlab "
            "task needs mjlab: pip install 'mjswan[mjlab]'."
        )
        raise typer.Exit(1)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1)
    ensure_mjlab_extensions()
    try:
        env_cfg = load_env_cfg(robot, play=True)
    except Exception as exc:
        console.print(
            f"[red]Error:[/red] {escape(robot)} is not an mjlab task: {escape(str(exc))}"
        )
        raise typer.Exit(1)
    env_cfg.scene.num_envs = 1
    return Scene(env_cfg.scene, device="cpu").spec.compile()


def _rest_pose(model, guess, bones: dict[str, str]):
    """The rest pose to draft, printed with its source; None, with the joints that set
    it, when it is not known."""
    from mjswan.rig import guess as g

    known = g.known_rest_pose(model)
    if known:
        console.print(
            f"  rest_pose  [dim]holds the arms out straight, checked for {known[0]}[/dim]"
        )
        return known[1]
    keys = [model.key(k).name for k in range(model.nkey)]
    if guess is not None and guess.kind == "quadruped" and keys and keys[0]:
        console.print(
            f'  rest_pose  "{escape(keys[0])}"  [dim]the robot\'s keyframe; it stands there[/dim]'
        )
        return keys[0]
    console.print(
        "  rest_pose  [dim]not guessed: set the joints that hold the pose the skin rests "
        f"in. Keyframes: {escape(', '.join(keys) or 'none')}.[/dim]"
    )
    for name, axis in g.arm_joints(model, bones):
        console.print(f"    {escape(name)}  [dim]axis {axis}[/dim]")
    return None


# ── version ───────────────────────────────────────────────────


@app.command("version")
def version_cmd() -> None:
    """Show the mjswan version."""
    console.print(f"mjswan {__version__}")


def _version_callback(value: bool) -> None:
    if value:
        version_cmd()
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    pass
