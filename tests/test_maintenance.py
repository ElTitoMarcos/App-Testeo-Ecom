import os
import subprocess
import unittest
from pathlib import Path

from product_research_app import database

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "product_research_app" / "data.sqlite3"


class PurgeResetTests(unittest.TestCase):
    def setUp(self) -> None:
        if DB_PATH.exists():
            DB_PATH.unlink()
        conn = database.get_connection(DB_PATH)
        database.initialize_database(conn)
        conn.close()

    def tearDown(self) -> None:
        if DB_PATH.exists():
            DB_PATH.unlink()
        backups = REPO_ROOT / "backups"
        if backups.exists():
            for f in backups.iterdir():
                f.unlink()

    def _run(self, extra_args=None, env=None):
        args = [
            "python",
            "-m",
            "product_research_app.maintenance",
            "purge-and-reset",
            "--yes",
            "--i-know-what-im-doing",
            "--no-prompt",
        ]
        if extra_args:
            args.extend(extra_args)
        env_vars = os.environ.copy()
        env_vars.update({"APP_ENV": "development"})
        if env:
            env_vars.update(env)
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            env=env_vars,
            capture_output=True,
            text=True,
        )

    def test_base_case_resets_id(self):
        conn = database.get_connection(DB_PATH)
        database.insert_product(conn, name="a")
        database.insert_product(conn, name="b")
        database.insert_product(conn, name="c")
        conn.close()
        res = self._run(extra_args=["--force"])
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        conn = database.get_connection(DB_PATH)
        new_id = database.insert_product(conn, name="nuevo")
        conn.close()
        self.assertEqual(new_id, 1)

    def test_idempotent(self):
        conn = database.get_connection(DB_PATH)
        database.insert_product(conn, name="x")
        conn.close()
        res1 = self._run(extra_args=["--force"])
        self.assertEqual(res1.returncode, 0)
        res2 = self._run(extra_args=["--force"])
        self.assertEqual(res2.returncode, 0)
        self.assertIn("nada que hacer", res2.stdout.lower())

    def test_foreign_keys(self):
        res = self._run()
        self.assertEqual(res.returncode, 1)
        self.assertIn("claves foráneas", res.stdout.lower())
        res_force = self._run(extra_args=["--force"])
        self.assertEqual(res_force.returncode, 0)

    def test_environment_guard(self):
        res = self._run(extra_args=["--force"], env={"APP_ENV": "production", "ENV": "production"})
        self.assertEqual(res.returncode, 1)


if __name__ == "__main__":
    unittest.main()
