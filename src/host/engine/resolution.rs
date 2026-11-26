//! Host-only registry and prepared-argument coverage for the engine's actual routes.
use std::collections::HashSet;

/// Resolve every registered STEP family and its members against the installed
/// registry before opening the API.
pub fn check() -> Result<(), Box<dyn std::error::Error>> {
    let rows = crate::host::registry::installed_rows()?;
    let installed: HashSet<_> = rows.into_iter().collect();
    let mut missing = Vec::new();
    let mut q4 = 0;
    for (name, row, paths) in super::step_image::resolution_catalog() {
        let mut routes = vec![row];
        for path in paths {
            routes.push(crate::host::prebuilt::launch_path(path));
            q4 += usize::from(path.contains("::q4::"));
        }
        for route in routes {
            if !installed.contains(&(route.to_string(), vec![])) {
                missing.push(format!("{name}: no device function {route} key=[]"));
            }
        }
    }
    if q4 == 0 {
        missing.push("Q4 enabled but no actual Q4 route in any STEP bucket".into());
    }
    if !missing.is_empty() {
        return Err(missing.join("\n").into());
    }
    Ok(())
}
