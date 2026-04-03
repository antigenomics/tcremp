import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tcremp.utils import resolve_prototype_file


class TestPrototypeResolution(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.resources_dir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_proto(self, name, rows):
        df = pd.DataFrame(rows)
        path = self.resources_dir / name
        df.to_csv(path, sep="\t", index=False)
        return path

    def _resource_lookup(self, name=None):
        if name is None:
            return sorted(p.name for p in self.resources_dir.iterdir())
        path = self.resources_dir / name
        if not path.exists():
            raise Exception("Missing resource")
        return str(path)

    def test_single_chain_prefers_split_resource_file(self):
        resource = self._write_proto(
            "tcremp_prototypes_IGH.tsv",
            [{"clone_id": "h1", "locus": "IGH", "v_call": "IGHV1", "j_call": "IGHJ1", "junction_aa": "CARDR"}],
        )

        with patch("tcremp.utils.get_resource_path", side_effect=self._resource_lookup):
            resolved = resolve_prototype_file(None, "IGH")

        self.assertEqual(Path(resolved), resource.resolve())

    def test_paired_chain_is_built_from_split_tra_trb_resources(self):
        self._write_proto(
            "tcremp_prototypes_TRA.tsv",
            [
                {"clone_id": "a1", "locus": "TRA", "v_call": "TRAV1", "j_call": "TRAJ1", "junction_aa": "CAVR"},
                {"clone_id": "a2", "locus": "TRA", "v_call": "TRAV2", "j_call": "TRAJ2", "junction_aa": "CALS"},
            ],
        )
        self._write_proto(
            "tcremp_prototypes_TRB.tsv",
            [
                {"clone_id": "b1", "locus": "TRB", "v_call": "TRBV1", "j_call": "TRBJ1", "junction_aa": "CASS"},
                {"clone_id": "b2", "locus": "TRB", "v_call": "TRBV2", "j_call": "TRBJ2", "junction_aa": "CASG"},
                {"clone_id": "b3", "locus": "TRB", "v_call": "TRBV3", "j_call": "TRBJ3", "junction_aa": "CAST"},
            ],
        )

        with patch("tcremp.utils.get_resource_path", side_effect=self._resource_lookup):
            resolved = Path(resolve_prototype_file(None, "TRA_TRB"))

        merged = pd.read_csv(resolved, sep="\t")
        self.assertEqual(len(merged), 4)
        self.assertEqual(sorted(merged["locus"].unique().tolist()), ["alpha", "beta"])
        self.assertEqual(merged["clone_id"].nunique(), 2)
        self.assertEqual(merged.groupby("clone_id").size().tolist(), [2, 2])

    def test_paired_chain_is_built_from_split_igh_igl_resources(self):
        self._write_proto(
            "tcremp_prototypes_IGH.tsv",
            [
                {"clone_id": "h1", "locus": "IGH", "v_call": "IGHV1", "j_call": "IGHJ1", "junction_aa": "CARDR"},
                {"clone_id": "h2", "locus": "IGH", "v_call": "IGHV2", "j_call": "IGHJ2", "junction_aa": "CARGG"},
            ],
        )
        self._write_proto(
            "tcremp_prototypes_IGL.tsv",
            [
                {"clone_id": "l1", "locus": "IGL", "v_call": "IGLV1", "j_call": "IGLJ1", "junction_aa": "CQNYD"},
                {"clone_id": "l2", "locus": "IGL", "v_call": "IGLV2", "j_call": "IGLJ2", "junction_aa": "CQHYG"},
                {"clone_id": "l3", "locus": "IGL", "v_call": "IGLV3", "j_call": "IGLJ3", "junction_aa": "CQSTS"},
            ],
        )

        with patch("tcremp.utils.get_resource_path", side_effect=self._resource_lookup):
            resolved = Path(resolve_prototype_file(None, "IGH_IGL"))

        merged = pd.read_csv(resolved, sep="\t")
        self.assertEqual(len(merged), 4)
        self.assertEqual(sorted(merged["locus"].unique().tolist()), ["igh", "igl"])
        self.assertEqual(merged["clone_id"].nunique(), 2)
        self.assertEqual(merged.groupby("clone_id").size().tolist(), [2, 2])

    def test_missing_split_resource_raises_clear_error(self):
        with patch("tcremp.utils.get_resource_path", side_effect=self._resource_lookup):
            with self.assertRaisesRegex(ValueError, "Expected split resource files"):
                resolve_prototype_file(None, "TRA")


if __name__ == "__main__":
    unittest.main()
