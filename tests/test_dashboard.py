"""Execute every tab and the lab interaction through Streamlit's runtime."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_tabs_and_controls(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(len(app.tabs), 4)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        app.toggle[0].set_value(True).run()
        app.slider[0].set_value(.5).run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        app.radio[0].set_value('Custom CSV Upload').run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        app.radio[1].set_value('Single-station spike').run()
        self.assertEqual(len(app.exception), 0, str(app.exception))


if __name__ == '__main__':
    unittest.main()
