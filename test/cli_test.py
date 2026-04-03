import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

try:
    import mir  # noqa: F401
    MIR_AVAILABLE = True
except ImportError:
    MIR_AVAILABLE = False

try:
    import kneed  # noqa: F401
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parents[1]
TCREMP_RUN_MODULE = "tcremp.tcremp_run"
TCREMP_CLUSTER_MODULE = "tcremp.tcremp_cluster"
TCREMP_ENRICH_MODULE = "tcremp.tcremp_enrich"
PROTOTYPES = REPO_ROOT / "test" / "test_data" / "tcremp_prototypes_olga_test.tsv"
EXAMPLE_INPUT = REPO_ROOT / "data" / "example" / "v_tcrpmhc.txt"


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = REPO_ROOT / "test" / ".tmp_cli"
        shutil.rmtree(cls.tmpdir, ignore_errors=True)
        cls.tmpdir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def run_cli(self, module_name, args, cwd=None):
        cmd = [sys.executable, "-m", module_name] + args
        result = subprocess.run(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"Command failed: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            ),
        )
        return result

    def make_input_file(self, chain, filename, max_clones=12):
        df = pd.read_csv(EXAMPLE_INPUT, sep="\t")

        if chain == "TRA":
            df = df[df["locus"] == "alpha"].copy().head(max_clones)
            df["test_label"] = df["junction_aa"].str.len().apply(lambda x: "long" if x >= 13 else "short")
        elif chain == "TRB":
            df = df[df["locus"] == "beta"].copy().head(max_clones)
            df["test_label"] = df["v_call"].str.contains("TRBV6").map({True: "TRBV6", False: "other"})
        elif chain == "TRA_TRB":
            clone_ids = df["clone_id"].drop_duplicates().head(max_clones).tolist()
            df = df[df["clone_id"].isin(clone_ids)].copy()
            label_map = {clone_id: ("even" if i % 2 == 0 else "odd") for i, clone_id in enumerate(clone_ids)}
            df["test_label"] = df["clone_id"].map(label_map)
        else:
            raise ValueError(f"Unsupported chain for test input: {chain}")

        input_path = self.tmpdir / filename
        df.to_csv(input_path, sep="\t", index=False)
        return input_path

    def make_output_dir(self, name):
        path = self.tmpdir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @unittest.skipUnless(MIR_AVAILABLE, "mir is required for tcremp-run integration tests")
    def test_tcremp_run_smoke_configs(self):
        configs = [
            {"chain": "TRA", "n_clonotypes": "10", "n_prototypes": "8", "include_index_col": False},
            {"chain": "TRB", "n_clonotypes": "10", "n_prototypes": "8", "include_index_col": True},
            {"chain": "TRA_TRB", "n_clonotypes": "6", "n_prototypes": "8", "include_index_col": True},
        ]

        for config in configs:
            chain = config["chain"]
            with self.subTest(chain=chain):
                input_path = self.make_input_file(chain, f"{chain.lower()}_input.tsv")
                output_dir = self.make_output_dir(f"run_{chain.lower()}")
                prefix = f"smoke_{chain.lower()}"

                args = [
                    "-i", str(input_path),
                    "-o", str(output_dir),
                    "-e", prefix,
                    "-c", chain,
                    "-p", str(PROTOTYPES),
                    "-n", config["n_prototypes"],
                    "-nc", config["n_clonotypes"],
                    "-np", "1",
                ]

                if config["include_index_col"]:
                    args += ["-x", "clone_id"]

                if chain == "TRB":
                    args += [
                        "--tsne",
                        "-l", "test_label",
                        "--enrich-by", "test_label",
                        "--enrichment-fdr-threshold", "1.0",
                    ]

                self.run_cli(TCREMP_RUN_MODULE, args)

                tcremp_path = output_dir / f"{prefix}_tcremp.parquet"
                clusters_path = output_dir / f"{prefix}_tcremp_clusters.tsv"
                reps_path = output_dir / f"{prefix}_tcremp_representations.tsv"

                self.assertTrue(tcremp_path.exists(), tcremp_path)
                self.assertTrue(clusters_path.exists(), clusters_path)
                self.assertTrue(reps_path.exists(), reps_path)

                tcremp_df = pd.read_parquet(tcremp_path)
                cluster_df = pd.read_csv(clusters_path, sep="\t")
                self.assertIn("cluster_id", cluster_df.columns)
                self.assertGreater(len(tcremp_df), 0)
                self.assertGreater(len(cluster_df), 0)
                self.assertFalse(any(col in tcremp_df.columns for col in ["v_call", "j_call", "junction_aa", "locus"]))
                if config["include_index_col"]:
                    self.assertIn("clone_id", tcremp_df.columns)
                else:
                    self.assertNotIn("clone_id", tcremp_df.columns)

                if chain == "TRB":
                    self.assertTrue((output_dir / f"{prefix}_tcremp_pca.tsv").exists())
                    self.assertTrue((output_dir / f"{prefix}_tcremp_tsne.tsv").exists())
                    self.assertTrue((output_dir / f"{prefix}_tcremp_tsne.png").exists())
                    self.assertTrue((output_dir / f"{prefix}_tcremp_enrichment_summary.tsv").exists())
                    self.assertTrue((output_dir / f"{prefix}_tcremp_clusters_enriched.tsv").exists())

                    enrichment_df = pd.read_csv(
                        output_dir / f"{prefix}_tcremp_enrichment_summary.tsv",
                        sep="\t",
                    )
                    self.assertIn("enrichment_fdr", enrichment_df.columns)
                    self.assertIn("enriched_cluster", enrichment_df.columns)
                    self.assertIn("passes_fraction_threshold", enrichment_df.columns)

    @unittest.skipUnless(KNEED_AVAILABLE, "kneed is required for tcremp-cluster CLI tests")
    def test_tcremp_cluster_cli(self):
        output_dir = self.make_output_dir("cluster_cli_run")
        numeric_df = pd.DataFrame(
            {
                "f1": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
                "f2": [0.0, 0.1, 0.2, 5.0, 5.1, 5.2],
                "f3": [1.0, 1.1, 1.2, 6.0, 6.1, 6.2],
            }
        )
        numeric_input = output_dir / "numeric_embeddings.tsv"
        numeric_df.to_csv(numeric_input, sep="\t", index=False)

        cluster_output = output_dir / "standalone_cluster.tsv"
        self.run_cli(
            TCREMP_CLUSTER_MODULE,
            [
                "--input", str(numeric_input),
                "--output", str(cluster_output),
                "--components", "5",
                "--min_samples", "2",
                "--kth_neighbor", "3",
            ],
        )

        clustered_df = pd.read_csv(cluster_output, sep="\t")
        self.assertIn("cluster", clustered_df.columns)
        self.assertEqual(len(clustered_df), len(numeric_df))

    def test_tcremp_enrich_cli(self):
        output_dir = self.make_output_dir("enrich_cli_run")
        cluster_df = pd.DataFrame(
            {
                "clone_id": [f"c{i}" for i in range(8)],
                "cluster_id": [0, 0, 0, 0, 1, 1, 1, -1],
                "test_label": ["A", "A", "A", "B", "B", "B", "B", "A"],
            }
        )
        cluster_file = output_dir / "synthetic_clusters.tsv"
        cluster_df.to_csv(cluster_file, sep="\t", index=False)

        standalone_output = self.make_output_dir("enrich_cli_standalone")
        standalone_prefix = "standalone_enrich"

        self.run_cli(
            TCREMP_ENRICH_MODULE,
            [
                "-i", str(cluster_file),
                "-o", str(standalone_output),
                "-e", standalone_prefix,
                "-l", "test_label",
                "--enrichment-fdr-threshold", "1.0",
            ],
        )

        summary_path = standalone_output / f"{standalone_prefix}_tcremp_enrichment_summary.tsv"
        enriched_clusters_path = standalone_output / f"{standalone_prefix}_tcremp_clusters_enriched.tsv"
        self.assertTrue(summary_path.exists(), summary_path)
        self.assertTrue(enriched_clusters_path.exists(), enriched_clusters_path)

        summary_df = pd.read_csv(summary_path, sep="\t")
        enriched_df = pd.read_csv(enriched_clusters_path, sep="\t")
        self.assertIn("label_cluster", summary_df.columns)
        self.assertIn("enrichment_pvalue", summary_df.columns)
        self.assertIn("enrichment_fdr", summary_df.columns)
        self.assertIn("enriched_cluster", summary_df.columns)
        self.assertIn("label_cluster", enriched_df.columns)


if __name__ == "__main__":
    unittest.main()
