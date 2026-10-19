# One Browser Analysis Per Patient And Block

User approved option A: manage only full browser workspace analyses, leaving
Notebook outputs and all raw ECoG/audio/baseline/recon inputs unchanged.

When selecting a patient/block with a saved browser analysis, offer Continue,
New And Replace, or Cancel. Without a saved browser analysis use a new draft.
Continue restores parameters, explicit trial review, accepted first/second
models, and display state. New prepares a fresh draft; replacement starts only
when the user submits extraction. No repeated overwrite question is needed.
Reloading an already attached analysis reconnects without prompting again.

One current analysis may contain both accepted rounds. Refitting replaces the
corresponding round, not an accumulating history of fits. Completed operations
prune obsolete browser snapshots/tasks and metadata revisions while retaining
files referenced by current numerical state, both current rounds, shared QC,
and active displays. Closing an old display precedes deletion of its files.

New extraction uses temporary staging. Only successful saved extraction becomes
current; failed/cancelled work must not destroy the previous resumable analysis.
Temporary staging is not a second historical analysis. Successful replacement
or explicit Continue consolidates prior recognized browser-only records for the
same context. Existing Notebook directories are never deleted or migrated.
All deletion targets must be contained verified browser-owned paths, with
links/junctions rejected. An interprocess context lease prevents two services
from overwriting the same analysis concurrently.

GET api/workspace/current returns {analysis: summary|null}; it never deletes.
POST start accepts replace_run_id, which must match the currently saved record
when one exists. A missing/stale acknowledgement rejects the request before
creating output. Existing open/state/action protocols otherwise remain.

Verify replacement success/failure/cancellation, resume, stale acknowledgements,
cross-context/Notebook preservation, link safety, exclusive leases, bounded
snapshots, and desktop/mobile dialogs. No numerical clustering changes, Notebook
edits, raw-data changes, or GitHub push are included.
