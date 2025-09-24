# Dependency Bootstrap Troubleshooting

The one-time dependency bootstrap flow stores state using either a marker file or a registry value. If application startup slows down or setup runs on every launch, review the following scenarios.

## Permission issues
- **Symptom:** The first run fails when creating the virtual environment or writing the setup marker.
- **Resolution:**
  - Run the script from an elevated command prompt if the installation paths require administrator access.
  - Confirm that the user has write access to the application directory (`logs`, `config`, and `.venv`).
  - Check `logs\setup_flag.log` or `logs\setup_registry.log` for the specific command that failed.

## Corrupted or deleted markers
- **Symptom:** Dependencies reinstall every time even though the initial setup succeeded.
- **Resolution:**
  - For the **flag file** workflow, ensure `config\setup_complete.flag` still exists and contains the timestamp and python path entries.
  - For the **registry** workflow, verify `HKCU\Software\EcomTestingApp` has the `SetupComplete` value. Remove any partially written or empty values before rerunning the script.
  - Re-run the script after restoring the marker; it will skip the heavy bootstrap when the marker is intact.

## Manual reset
- **Symptom:** You intentionally need to rerun the dependency bootstrap (e.g., upgrading Python or dependencies).
- **Resolution:**
  - Delete `config\setup_complete.flag` for the file-based approach or run `reg delete "HKCU\Software\EcomTestingApp" /v "SetupComplete" /f` for the registry approach.
  - On the next launch, the script will perform the dependency verification and recreate the markers upon success.

## Network or download failures
- **Symptom:** Python or dependency downloads fail during the first run.
- **Resolution:**
  - Verify network connectivity and proxy settings used by `Invoke-WebRequest`.
  - Inspect the setup log for the HTTP status code or PowerShell error. Retry after the connection is restored.

Maintain the log files (`logs\setup_flag.log` and `logs\setup_registry.log`) for auditing. They capture every bootstrap attempt and highlight the commands to retry manually when necessary.
