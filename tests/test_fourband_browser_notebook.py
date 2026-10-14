"""Execute browser display sources and migrate user notebooks without data loss."""
import ast
from copy import deepcopy
import shutil
import subprocess
import sys

import nbformat
import pytest

from test_fourband_continuous_notebook import ROOT, generator
from test_fourband_display_notebook import cell_source, display_namespace


def backend_settings(notebook, namespace):
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assignments = [node for node in ast.parse(cell.source).body
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "DISPLAY_BACKEND"
                for target in node.targets)]
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "settings", "exec"), namespace)


@pytest.fixture(params=["current", "reliability", "legacy"])
def notebook(request):
    if request.param == "current":
        return nbformat.read(ROOT / "3-bands/ECoG_4band_clustering_pipeline.ipynb", as_version=4)
    builder = generator()
    return builder.build_notebook() if request.param == "legacy" else builder.build_reliability_notebook()


def launch_source(notebook, marker, tmp_path, backend=None):
    namespace, calls, display_run = display_namespace(tmp_path)
    namespace.pop("DISPLAY_BACKEND")
    backend_settings(notebook, namespace)
    if backend is not None:
        namespace["DISPLAY_BACKEND"] = backend
    launch = namespace["run_display"]

    def record_backend(*args, backend="missing", **kwargs):
        result = launch(*args, **kwargs)
        calls[-1]["backend"] = backend
        return result

    namespace["run_display"] = record_backend
    original_labels = deepcopy((namespace["first_result"].labels, namespace["second_result"].labels))
    exec(cell_source(notebook, marker), namespace)
    assert (namespace["first_result"].labels, namespace["second_result"].labels) == original_labels
    assert calls[0]["result"] is namespace["second_result"]
    assert namespace["surface_display_run" if marker == "if RUN_3D_RENDER" else "explorer_display_run"] is display_run
    return calls


@pytest.mark.parametrize("marker", ["if RUN_3D_RENDER", "if OPEN_EXPLORER:"])
def test_display_defaults_to_browser_without_parent_data_loading(notebook, marker, tmp_path):
    if marker == "if OPEN_EXPLORER:" and not any(marker in c.source for c in notebook.cells):
        pytest.skip("Base notebook has no standalone explorer")
    calls = launch_source(notebook, marker, tmp_path)
    assert len(calls) == 1
    assert calls[0]["backend"] == "browser"


@pytest.mark.parametrize("marker", ["if RUN_3D_RENDER", "if OPEN_EXPLORER:"])
def test_explicit_native_backend_is_forwarded_to_launcher(tmp_path, marker):
    calls = launch_source(generator().build_reliability_notebook(), marker, tmp_path, "native")
    assert calls[0]["backend"] == "native"


def user_notebook(backend):
    notebook = generator().build_reliability_notebook()
    # Emulate the saved pre-browser notebook, including user edits in owned cells.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = "\n".join(line for line in cell.source.splitlines()
                if not line.startswith("DISPLAY_BACKEND ="))
            cell.source = cell.source.replace(", backend=DISPLAY_BACKEND", "")
            cell.source += "\nuser_annotation = 'retain this edit'"
            cell.execution_count = 19
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="saved output\n")]
        cell.metadata["user_note"] = "keep"
    params = next(c for c in notebook.cells if "def extraction_settings():" in c.source)
    params.source = params.source.replace('SUBJECT_ID = "HS0023"', 'SUBJECT_ID = "HS0015"').replace(
        'BLOCK_ID = "B07"', 'BLOCK_ID = "B03"').replace("FIRST_N_CLUSTERS = 10", "FIRST_N_CLUSTERS = 7")
    if backend is not None:
        params.source += f'\nDISPLAY_BACKEND = "{backend}"'
    notebook.cells.insert(8, nbformat.v4.new_markdown_cell("User analysis notes"))
    notebook.metadata["user_metadata"] = {"preserve": True}
    return notebook


def assert_source_only_browser_update(before, after):
    assert len(after.cells) == len(before.cells)
    restored = deepcopy(after)
    for old, new in zip(before.cells, restored.cells):
        new.source = old.source
    assert restored == before  # Includes outputs, counts, IDs, metadata and cell order.
    for old, new in zip(before.cells, after.cells):
        expected = old.source
        if old.cell_type == "code" and "def extraction_settings():" in old.source and "DISPLAY_BACKEND =" not in old.source:
            expected += '\nDISPLAY_BACKEND = "browser"'
        if old.cell_type == "code" and "run_display(" in old.source:
            # Only the backend keyword may change in existing display code.
            assert new.source.replace(", backend=DISPLAY_BACKEND", "") == expected
        else:
            assert new.source == expected
    nbformat.validate(after)
    for cell in after.cells:
        if cell.cell_type == "code":
            compile(cell.source, "migrated cell", "exec")


@pytest.mark.parametrize("migration", ["update_browser_cells", "update_review_cells", "update_continuous_cells", "update_reliability_cells"])
@pytest.mark.parametrize("backend", [None, "browser", "native"])
def test_migrations_preserve_all_user_content_and_are_idempotent(tmp_path, migration, backend):
    notebook = user_notebook(backend)
    path = tmp_path / "user.ipynb"
    nbformat.write(notebook, path)
    update = getattr(generator(), migration)
    update(path)
    updated = nbformat.read(path, as_version=4)
    assert_source_only_browser_update(notebook, updated)
    assert launch_source(updated, "if RUN_3D_RENDER", tmp_path)[0]["backend"] == (backend or "browser")
    assert launch_source(updated, "if OPEN_EXPLORER:", tmp_path)[0]["backend"] == (backend or "browser")
    once = path.read_bytes()
    update(path)
    assert path.read_bytes() == once


def test_generator_default_does_not_regenerate_existing_user_notebook(tmp_path):
    script = tmp_path / "create_notebook.py"
    shutil.copyfile(ROOT / "3-bands/create_notebook.py", script)
    path = tmp_path / "ECoG_4band_clustering_pipeline.ipynb"
    before = user_notebook(None)
    nbformat.write(before, path)
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True)
    assert_source_only_browser_update(before, nbformat.read(path, as_version=4))


@pytest.mark.parametrize("marker", ["if RUN_3D_RENDER", "if OPEN_EXPLORER:"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_launcher_failure_propagates_without_retry_or_classification_changes(tmp_path, marker, error):
    notebook = generator().build_reliability_notebook()
    namespace, _, _ = display_namespace(tmp_path)
    backend_settings(notebook, namespace)
    results = (namespace["first_result"], namespace["second_result"])
    labels = deepcopy([result.labels for result in results])
    attempts = []

    def fail(*args, backend, **kwargs):
        attempts.append(backend)
        raise error("display failure")

    namespace["run_display"] = fail
    with pytest.raises(error, match="display failure"):
        exec(cell_source(notebook, marker), namespace)
    assert attempts == ["browser"]
    assert [result.labels for result in results] == labels
    assert namespace["SECOND_ROUND_READY"] is True


@pytest.mark.parametrize("marker", ["def extraction_settings():", "if RUN_3D_RENDER", "if OPEN_EXPLORER:"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_browser_migration_rejects_unowned_cells_without_writing(tmp_path, marker, ambiguous):
    notebook = user_notebook(None)
    owned = next(cell for cell in notebook.cells if marker in cell.source)
    if ambiguous:
        notebook.cells.append(nbformat.v4.new_code_cell(owned.source))
    else:
        notebook.cells.remove(owned)
    path = tmp_path / "unowned.ipynb"
    nbformat.write(notebook, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Cannot uniquely identify owned notebook cell"):
        generator().update_browser_cells(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("migration", ["update_review_cells", "update_continuous_cells", "update_reliability_cells"])
def test_legacy_direct_display_migration_adds_browser_and_preserves_saved_outputs(tmp_path, migration):
    notebook = user_notebook(None)
    surface = next(c for c in notebook.cells if "if RUN_3D_RENDER" in c.source)
    surface.source = "if RUN_3D_RENDER and SECOND_ROUND_READY:\n    render_electrode_clusters(ELECTRODE_CSV, {})"
    explorer = next(c for c in notebook.cells if "if OPEN_EXPLORER:" in c.source)
    explorer.source = 'if OPEN_EXPLORER:\n    get_explorer("second", second_result, ROUND2_DIR).show(block=True)'
    parameters = cell_source(notebook, "def extraction_settings():")
    path = tmp_path / "direct.ipynb"
    nbformat.write(notebook, path)
    update = getattr(generator(), migration)
    update(path)
    updated = nbformat.read(path, as_version=4)
    assert cell_source(updated, "def extraction_settings():") == parameters + '\nDISPLAY_BACKEND = "browser"'
    for before, after in zip(notebook.cells, updated.cells):
        assert after.id == before.id
        assert after.metadata == before.metadata
        if before.cell_type == "code":
            assert after.outputs == before.outputs
            assert after.execution_count == before.execution_count
    for marker in ("if RUN_3D_RENDER", "if OPEN_EXPLORER:"):
        assert launch_source(updated, marker, tmp_path)[0]["backend"] == "browser"
    once = path.read_bytes()
    update(path)
    assert path.read_bytes() == once
