"""Tests for Telemetry Studio and Microgravity Combustion Telemetry Engine."""
import unittest
from src.visualisation.telemetry_studio import TelemetryStudio


class TestTelemetryStudio(unittest.TestCase):
    """Verifies that the merged flame_project telemetry engine functions reliably."""

    def test_available_datasets(self):
        datasets = TelemetryStudio.get_available_datasets()
        self.assertIsInstance(datasets, list)
        self.assertIn("BASS-II", datasets)
        self.assertIn("FLEX", datasets)

    def test_bundle_generation_bass_ii(self):
        bundle = TelemetryStudio.get_telemetry_bundle(
            dataset="BASS-II",
            plot="01_scatter_outcomes",
            bins=8,
            min_samples=2,
            method="uniform",
        )
        self.assertIn("dataset_name", bundle)
        self.assertEqual(bundle["total_rows"], 123)
        self.assertIn("plots", bundle)
        self.assertIn("01_scatter_outcomes", bundle["plots"])
        plot_info = bundle["plots"]["01_scatter_outcomes"]
        self.assertTrue(plot_info["image"].startswith("data:image/png;base64,"))
        self.assertIn("Maps individual microgravity test points", plot_info["interpretation"])
        self.assertIn("tables", bundle)
        self.assertIn("summary", bundle["tables"])

    def test_bundle_cache_acceleration(self):
        # First call populates cache
        b1 = TelemetryStudio.get_telemetry_bundle(dataset="BASS-II", plot="01_scatter_outcomes")
        # Second call hits cache
        b2 = TelemetryStudio.get_telemetry_bundle(dataset="BASS-II", plot="01_scatter_outcomes")
        self.assertEqual(b1["total_rows"], b2["total_rows"])
        self.assertEqual(
            b1["plots"]["01_scatter_outcomes"]["image"][:50],
            b2["plots"]["01_scatter_outcomes"]["image"][:50],
        )

    def test_bundle_generation_flex(self):
        bundle = TelemetryStudio.get_telemetry_bundle(
            dataset="FLEX",
            plot="02_sustain_rate_heatmap",
            bins=6,
            min_samples=2,
        )
        self.assertIn("dataset_name", bundle)
        self.assertEqual(bundle["total_rows"], 274)
        self.assertIn("plots", bundle)
        self.assertIn("02_sustain_rate_heatmap", bundle["plots"])
        plot_info = bundle["plots"]["02_sustain_rate_heatmap"]
        self.assertTrue(plot_info["image"].startswith("data:image/png;base64,"))
        self.assertIn("Two-dimensional discretization", plot_info["interpretation"])

    def test_parametric_burn_risk_factor(self):
        bundle = TelemetryStudio.get_telemetry_bundle(
            dataset="BASS-II",
            plot="08_burn_risk_surface",
            bins=8,
            min_samples=2,
        )
        self.assertIn("risk_summary", bundle)
        risk = bundle["risk_summary"]
        self.assertIn("overall_burn_risk", risk)
        self.assertIn("oxygen_risk_sensitivity", risk)
        self.assertIn("peak_hazard_flow_band", risk)
        self.assertIn("risk_tier", risk)
        self.assertIn("08_burn_risk_surface", bundle["plots"])
        plot_info = bundle["plots"]["08_burn_risk_surface"]
        self.assertTrue(plot_info["image"].startswith("data:image/png;base64,"))
        self.assertIn("burn risk", plot_info["interpretation"].lower())


if __name__ == "__main__":
    unittest.main()
