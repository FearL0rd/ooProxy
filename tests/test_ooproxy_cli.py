"""Unit tests for the ooproxy.py CLI host.

Covers:
  - invalid module discovery (missing SPEC / run / render_text)
  - flag conflict between two modules sharing the same option with different specs
  - required-option validation (missing required flag → error, exit 2)
  - JSON output rendering (--json flag)
  - text rendering (default, no --json flag)
  - render_text is safe even when module never gets assigned (early CommandError)
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from io import StringIO
from unittest.mock import patch

from cli_contract import (
    CommandError,
    ModuleSpec,
    OptionSpec,
    ResultEnvelope,
    command_result,
)
from ooproxy import (
    build_parser,
    discover_modules,
    main,
    print_result,
    validate_required_options,
)


# ---------------------------------------------------------------------------
# Helpers — build minimal fake modules
# ---------------------------------------------------------------------------

def _make_module(
    name: str,
    *,
    spec=True,
    run=True,
    render_text=True,
    options: tuple[OptionSpec, ...] = (),
    examples: tuple[str, ...] = (),
    run_return=None,
    render_return: str = "",
) -> types.ModuleType:
    """Return a fake module object that satisfies (or intentionally violates) the contract."""
    mod = types.ModuleType(name)
    if spec:
        mod.SPEC = ModuleSpec(
            name=name,
            action_flags=(f"-{name[0]}", f"--{name}"),
            help=f"Help for {name}",
            options=options,
            usage_examples=examples,
        )
    if run:
        def _run(args):
            if run_return is not None:
                return run_return
            return command_result(name, None, data={"ok": True})
        mod.run = _run
    if render_text:
        mod.render_text = lambda result: render_return
    return mod


# ---------------------------------------------------------------------------
# 1. Invalid module discovery
# ---------------------------------------------------------------------------

class TestDiscoverModules(unittest.TestCase):

    def _discover(self, modules_dict: dict):
        """Patch pkgutil + importlib to present fake modules and call discover_modules()."""
        import pkgutil

        fake_infos = [
            types.SimpleNamespace(name=name)
            for name in modules_dict
        ]

        def fake_iter_modules(path):
            return iter(fake_infos)

        def fake_import_module(fullname):
            # Return top-level package for the MODULE_PACKAGE itself, or the fake module
            parts = fullname.split(".")
            if len(parts) == 1:
                pkg = types.ModuleType(fullname)
                pkg.__path__ = []
                return pkg
            return modules_dict[parts[-1]]

        with patch("pkgutil.iter_modules", fake_iter_modules), \
             patch("importlib.import_module", fake_import_module):
            return discover_modules()

    def test_valid_module_is_accepted(self):
        mod = _make_module("serve")
        result = self._discover({"serve": mod})
        self.assertIn("serve", result)

    def test_missing_spec_raises(self):
        mod = _make_module("serve", spec=False)
        with self.assertRaises(CommandError):
            self._discover({"serve": mod})

    def test_missing_run_raises(self):
        mod = _make_module("serve", run=False)
        with self.assertRaises(CommandError):
            self._discover({"serve": mod})

    def test_missing_render_text_raises(self):
        mod = _make_module("serve", render_text=False)
        with self.assertRaises(CommandError):
            self._discover({"serve": mod})

    def test_no_modules_raises(self):
        """discover_modules() with an empty package must raise CommandError."""
        import pkgutil
        pkg = types.ModuleType("modules")
        pkg.__path__ = []

        with patch("pkgutil.iter_modules", lambda path: iter([])), \
             patch("importlib.import_module", lambda n: pkg):
            with self.assertRaises(CommandError):
                discover_modules()


# ---------------------------------------------------------------------------
# 2. Flag conflict
# ---------------------------------------------------------------------------

class TestFlagConflict(unittest.TestCase):

    def test_conflicting_option_raises(self):
        """Two modules declare the same flag with different dest → CommandError."""
        opt_a = OptionSpec(flags=("--url",), dest="url", help="URL for A", metavar="URL")
        opt_b = OptionSpec(flags=("--url",), dest="endpoint", help="URL for B", metavar="URL")

        mod_a = _make_module("alpha", options=(opt_a,))
        mod_b = _make_module("beta",  options=(opt_b,))

        with self.assertRaises(CommandError):
            build_parser({"alpha": mod_a, "beta": mod_b})

    def test_identical_shared_option_is_accepted(self):
        """Same flags with identical spec on two modules must not raise."""
        opt = OptionSpec(flags=("--url",), dest="url", help="URL", metavar="URL")
        mod_a = _make_module("alpha", options=(opt,))
        mod_b = _make_module("beta",  options=(opt,))
        # Should not raise
        parser, _ = build_parser({"alpha": mod_a, "beta": mod_b})
        self.assertIsNotNone(parser)


# ---------------------------------------------------------------------------
# 3. Required-option validation
# ---------------------------------------------------------------------------

class TestRequiredOptions(unittest.TestCase):

    def _parse(self, argv: list[str], modules: dict):
        parser, _ = build_parser(modules)
        return parser.parse_args(argv)

    def test_missing_required_option_raises(self):
        req = OptionSpec(flags=("--key",), dest="key", help="API key", required=True)
        mod = _make_module("serve", options=(req,))
        args = self._parse(["--serve"], {"serve": mod})
        with self.assertRaises(CommandError) as ctx:
            validate_required_options(args, mod.SPEC)
        self.assertIn("--key", str(ctx.exception))

    def test_present_required_option_passes(self):
        req = OptionSpec(flags=("--key",), dest="key", help="API key", required=True)
        mod = _make_module("serve", options=(req,))
        args = self._parse(["--serve", "--key", "sk-test"], {"serve": mod})
        # Must not raise
        validate_required_options(args, mod.SPEC)

    def test_main_exits_with_code_2_on_missing_required(self):
        """End-to-end: main() returns exit code 2 for a missing required option."""
        req = OptionSpec(flags=("--key",), dest="key", help="API key", required=True, metavar="KEY")
        run_sentinel = command_result("serve", None, data=None)
        mod = _make_module("serve", options=(req,), run_return=run_sentinel)

        def fake_discover():
            return {"serve": mod}

        with patch("ooproxy.discover_modules", fake_discover), \
             patch("sys.stderr", StringIO()):
            rc = main(["--serve"])
        self.assertEqual(rc, 2)


# ---------------------------------------------------------------------------
# 4. JSON rendering
# ---------------------------------------------------------------------------

class TestJsonRendering(unittest.TestCase):

    def test_json_flag_produces_valid_json(self):
        result = command_result("serve", None, data={"key": "value"})
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc = print_result(result, json_output=True, render_text=lambda r: "ignored")
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["data"]["key"], "value")
        self.assertEqual(parsed["status"], "ok")

    def test_json_output_contains_all_envelope_fields(self):
        result = ResultEnvelope(
            command="test",
            source="src",
            status="ok",
            data={"x": 1},
            warnings=["w1"],
            errors=[],
        )
        buf = StringIO()
        with patch("sys.stdout", buf):
            print_result(result, json_output=True, render_text=lambda r: "")
        parsed = json.loads(buf.getvalue())
        self.assertIn("command", parsed)
        self.assertIn("warnings", parsed)
        self.assertIn("errors", parsed)

    def test_error_status_without_json_returns_1(self):
        result = ResultEnvelope(
            command="serve", source=None, status="error",
            data=None, warnings=[], errors=["something went wrong"],
        )
        buf = StringIO()
        with patch("sys.stderr", buf):
            rc = print_result(result, json_output=False, render_text=lambda r: "")
        self.assertEqual(rc, 1)
        self.assertIn("something went wrong", buf.getvalue())


# ---------------------------------------------------------------------------
# 5. Text rendering
# ---------------------------------------------------------------------------

class TestTextRendering(unittest.TestCase):

    def test_render_text_output_is_printed(self):
        result = command_result("serve", None, data={"items": [1, 2, 3]})
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc = print_result(result, json_output=False, render_text=lambda r: "rendered output")
        self.assertEqual(rc, 0)
        self.assertIn("rendered output", buf.getvalue())

    def test_empty_render_text_prints_nothing(self):
        result = command_result("serve", None, data={"items": []})
        buf = StringIO()
        with patch("sys.stdout", buf):
            rc = print_result(result, json_output=False, render_text=lambda r: "")
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue(), "")

    def test_render_text_not_called_when_data_is_none(self):
        called = []
        result = command_result("serve", None, data=None)
        buf_out = StringIO()
        buf_err = StringIO()
        with patch("sys.stdout", buf_out), patch("sys.stderr", buf_err):
            print_result(result, json_output=False, render_text=lambda r: called.append(1) or "x")
        # render_text must not be invoked when data is None and status is ok
        self.assertEqual(called, [])

    def test_render_text_safe_before_module_assigned(self):
        """Simulate an early CommandError (before module is assigned).

        main() must not raise NameError — render_text fallback keeps it safe.
        """
        def exploding_discover():
            raise CommandError("no modules found")

        buf = StringIO()
        with patch("ooproxy.discover_modules", exploding_discover), \
             patch("sys.stderr", buf):
            rc = main(["--serve"])
        self.assertEqual(rc, 1)
        self.assertIn("no modules found", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
