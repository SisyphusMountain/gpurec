//! Shared Python types for reconciliation statistics.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Event counts from reconciliation analysis
#[pyclass]
#[derive(Clone)]
pub struct PyEventCounts {
    /// Number of speciation events
    #[pyo3(get)]
    pub speciations: f64,

    /// Number of speciation loss events
    #[pyo3(get)]
    pub speciation_losses: f64,

    /// Number of duplication events
    #[pyo3(get)]
    pub duplications: f64,

    /// Number of duplication loss events
    #[pyo3(get)]
    pub duplication_losses: f64,

    /// Number of transfer events
    #[pyo3(get)]
    pub transfers: f64,

    /// Number of transfer loss events
    #[pyo3(get)]
    pub transfer_losses: f64,

    /// Number of loss events
    #[pyo3(get)]
    pub losses: f64,

    /// Number of leaf nodes
    #[pyo3(get)]
    pub leaves: f64,
}

#[pymethods]
impl PyEventCounts {
    fn __repr__(&self) -> String {
        format!(
            "EventCounts(S={:.1}, D={:.1}, T={:.1}, L={:.1})",
            self.speciations,
            self.duplications,
            self.transfers,
            self.losses
        )
    }
}

/// Summary statistics for reconciliation analysis
#[pyclass]
#[derive(Clone)]
pub struct PyReconciliationStatistics {
    /// Mean event counts across all samples
    #[pyo3(get)]
    pub mean_event_counts: PyEventCounts,

    /// Mean transfers between species pairs
    /// Returns a dict of dicts: {source_species: {dest_species: mean_count}}
    #[pyo3(get)]
    pub mean_transfers: HashMap<String, HashMap<String, f64>>,

    /// Mean events per species node
    /// Returns a dict: {species_name: EventCounts}
    #[pyo3(get)]
    pub events_per_species: HashMap<String, PyEventCounts>,
}

#[pymethods]
impl PyReconciliationStatistics {
    fn __repr__(&self) -> String {
        let total_transfers: f64 = self.mean_transfers.values()
            .flat_map(|dests| dests.values())
            .sum();
        format!(
            "ReconciliationStatistics(mean_events={}, mean_transfers={:.1}, species_count={})",
            self.mean_event_counts.__repr__(),
            total_transfers,
            self.events_per_species.len()
        )
    }

    /// Returns a pandas DataFrame of mean transfers between species pairs.
    ///
    /// Columns: source, destination, mean_count
    fn transfers_df(&self, py: Python) -> PyResult<PyObject> {
        let pandas = super::import_pymodule(py, "pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let mut sources: Vec<String> = Vec::new();
        let mut destinations: Vec<String> = Vec::new();
        let mut counts: Vec<f64> = Vec::new();

        for (source, dests) in &self.mean_transfers {
            for (dest, count) in dests {
                sources.push(source.clone());
                destinations.push(dest.clone());
                counts.push(*count);
            }
        }

        dict.set_item("source", sources)?;
        dict.set_item("destination", destinations)?;
        dict.set_item("mean_count", counts)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }

    /// Returns a pandas DataFrame of mean events per species.
    ///
    /// Columns: species, speciations, speciation_losses, duplications,
    ///          duplication_losses, transfers, transfer_losses, losses, leaves
    fn events_df(&self, py: Python) -> PyResult<PyObject> {
        let pandas = super::import_pymodule(py, "pandas")?;
        let dict = pyo3::types::PyDict::new(py);

        let mut species_names: Vec<String> = Vec::new();
        let mut speciations: Vec<f64> = Vec::new();
        let mut speciation_losses: Vec<f64> = Vec::new();
        let mut duplications: Vec<f64> = Vec::new();
        let mut duplication_losses: Vec<f64> = Vec::new();
        let mut transfers: Vec<f64> = Vec::new();
        let mut transfer_losses: Vec<f64> = Vec::new();
        let mut losses: Vec<f64> = Vec::new();
        let mut leaves: Vec<f64> = Vec::new();

        // Sort by species name for consistent output
        let mut sorted_species: Vec<(&String, &PyEventCounts)> =
            self.events_per_species.iter().collect();
        sorted_species.sort_by(|(a, _), (b, _)| a.cmp(b));

        for (species, counts) in sorted_species {
            species_names.push(species.clone());
            speciations.push(counts.speciations);
            speciation_losses.push(counts.speciation_losses);
            duplications.push(counts.duplications);
            duplication_losses.push(counts.duplication_losses);
            transfers.push(counts.transfers);
            transfer_losses.push(counts.transfer_losses);
            losses.push(counts.losses);
            leaves.push(counts.leaves);
        }

        dict.set_item("species", species_names)?;
        dict.set_item("speciations", speciations)?;
        dict.set_item("speciation_losses", speciation_losses)?;
        dict.set_item("duplications", duplications)?;
        dict.set_item("duplication_losses", duplication_losses)?;
        dict.set_item("transfers", transfers)?;
        dict.set_item("transfer_losses", transfer_losses)?;
        dict.set_item("losses", losses)?;
        dict.set_item("leaves", leaves)?;

        let df = pandas.call_method1("DataFrame", (dict,))?;
        Ok(df.into())
    }
}
