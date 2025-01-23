import subprocess
import pandas as pd
import unittest

CLI_APP_PATH = "../tcremp/tcremp_run.py"


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Читаем параметры из файла перед запуском тестов."""
        df = pd.read_csv("flags.csv", delimiter='\t')
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].astype('Int64')
        cls.test_cases = [
            " ".join(
                f"--{col.replace('_', '-')} {val}" for col, val in row.items() if pd.notna(val) and str(val).lower() != "none") for
            _, row in df.iterrows()]

    def test_cli_exit_code(self):
        """Запускает CLI с различными аргументами и проверяет, что код выхода 0."""
        for args in self.test_cases:
            with self.subTest(args=args):
                cmd = ["python", CLI_APP_PATH] + args.split()
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.assertEqual(result.returncode, 0, f"Failed for args: {args}\nError: {result.stderr.decode()}")


if __name__ == "__main__":
    unittest.main()
