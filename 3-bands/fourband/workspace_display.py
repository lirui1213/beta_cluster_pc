"""Prepare anatomy and metadata in a disposable, owned numerical worker."""
from pathlib import Path


def validate_source_identity(sources, identity):
    """Catch replaced recordings/events/audio before mixing them with saved labels."""
    for name, expected in (identity or {}).items():
        path = Path(sources[name])
        try:
            stat = path.stat()
        except OSError as error:
            raise ValueError(f"Saved source input {name} is unavailable; start a new analysis when restored.") from error
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Saved input {name} changed; start a new analysis instead of mixing it with old labels.")


def guarded_display_call(sources, identity, function, args):
    validate_source_identity(sources, identity)
    return function(*args)


def prepare_display(request_path, run_dir):
    path, root = Path(request_path).resolve(), Path(run_dir).resolve()
    if not path.is_relative_to(root) or path.name != "request.joblib":
        raise ValueError("Display request must be a trusted snapshot inside the accepted run.")
    import joblib
    from .browser_data import session_metadata, surface_payload
    request = joblib.load(path)
    validate_source_identity(request["sources"], request.get("input_identity"))
    metadata = session_metadata(request)
    metadata["warnings"].extend(request.get("warnings", []))
    surface = None
    if metadata["surface_enabled"]:
        try:
            surface = surface_payload(request)
        except Exception as error:
            metadata["surface_enabled"] = False
            metadata["warnings"].append(str(error))
    return dict(metadata=metadata, surface=surface, output_dir=str(Path(request["output_dir"]).resolve()),
                sources=request["sources"], input_identity=request.get("input_identity"))


def guarded_computation(request_path, sources, identity):
    """Keep the legacy explorer intact while guarding this workspace's workers."""
    from .display_computation import DisplayComputation, prepare_snapshot

    class GuardedComputation(DisplayComputation):
        def submit(self, settings):
            return self.submit_task(prepare_snapshot, self.request_path, settings)

        def submit_task(self, function, *args):
            return super().submit_task(guarded_display_call, sources, identity, function, args)

    return GuardedComputation(request_path)
