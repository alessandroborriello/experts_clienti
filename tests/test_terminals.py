import tempfile
import unittest
from pathlib import Path

from engine.terminals import discover_terminals


def _make_terminal(root: Path, data_id: str, install_dir: Path, login: str = None,
                    server: str = None, make_exe: bool = True, utf16: bool = False) -> None:
    data_dir = root / data_id
    data_dir.mkdir(parents=True, exist_ok=True)

    install_dir.mkdir(parents=True, exist_ok=True)
    if make_exe:
        (install_dir / "terminal64.exe").write_bytes(b"MZ")   # contenuto finto, basta che esista

    text = str(install_dir)
    encoding = "utf-16" if utf16 else "utf-8"
    (data_dir / "origin.txt").write_text(text, encoding=encoding)

    (data_dir / "config").mkdir(exist_ok=True)
    ini_lines = []
    if login:
        ini_lines.append(f"Login={login}")
    if server:
        ini_lines.append(f"Server={server}")
    (data_dir / "config" / "common.ini").write_text("\n".join(ini_lines), encoding=encoding)


class TestDiscoverTerminals(unittest.TestCase):
    def test_finds_valid_terminal_with_login_and_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Terminal"
            install = Path(tmp) / "Install" / "MetaTraderNuova"
            _make_terminal(root, "AAA111", install, login="7004626", server="KeyToMarkets-Server")

            found = discover_terminals(str(root))

            self.assertEqual(len(found), 1)
            t = found[0]
            self.assertEqual(t.id, "AAA111")
            self.assertEqual(t.name, "MetaTraderNuova")
            self.assertEqual(t.login, "7004626")
            self.assertEqual(t.server, "KeyToMarkets-Server")
            self.assertTrue(t.exe.endswith("terminal64.exe"))
            self.assertFalse(t.running)   # nessun processo vero in esecuzione nel test

    def test_skips_folder_without_terminal64_exe(self):
        # Simula un terminale MQL4 (solo terminal.exe, non terminal64.exe) o un'installazione rimossa
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Terminal"
            install = Path(tmp) / "Install" / "MT4Vecchio"
            _make_terminal(root, "BBB222", install, make_exe=False)

            found = discover_terminals(str(root))
            self.assertEqual(found, [])

    def test_skips_folder_without_origin_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Terminal"
            (root / "CCC333").mkdir(parents=True)   # cartella dati vuota, nessun origin.txt

            found = discover_terminals(str(root))
            self.assertEqual(found, [])

    def test_handles_utf16_config_files(self):
        # I file di MetaTrader sono spesso UTF-16 con BOM, come i sorgenti .mq5
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Terminal"
            install = Path(tmp) / "Install" / "MetaTrader"
            _make_terminal(root, "DDD444", install, login="7001023",
                            server="KeyToMarkets-Server", utf16=True)

            found = discover_terminals(str(root))
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].login, "7001023")

    def test_terminal_without_login_yet(self):
        # Installato ma mai collegato: niente Login=/Server= nel common.ini
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Terminal"
            install = Path(tmp) / "Install" / "MetaTrader5"
            _make_terminal(root, "EEE555", install)

            found = discover_terminals(str(root))
            self.assertEqual(len(found), 1)
            self.assertIsNone(found[0].login)
            self.assertIsNone(found[0].server)

    def test_multiple_terminals_sorted_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Terminal"
            _make_terminal(root, "ZZZ999", Path(tmp) / "Install" / "Z", login="1")
            _make_terminal(root, "AAA111", Path(tmp) / "Install" / "A", login="2")

            found = discover_terminals(str(root))
            self.assertEqual([t.id for t in found], ["AAA111", "ZZZ999"])

    def test_empty_root_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            found = discover_terminals(str(Path(tmp) / "nonexistent"))
            self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
