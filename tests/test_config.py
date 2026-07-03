"""Section 2 — Config (dfbench.core.config).

Tests 2.1–2.4: create_parser behaviour.
"""

from dfbench.core.config import create_parser


SAMPLE_PARAMS = {
    "pop_size": 100,
    "learning_rate": 0.01,
    "algorithm_name": "adam",
    "use_cuda": False,
    "verbose": True,
}


class TestCreateParser:
    """2.1–2.4"""

    def test_snake_to_kebab(self):
        """2.1 snake_case → --kebab-case."""
        parser = create_parser({"pop_size": 100})
        # Should accept --pop-size
        args = parser.parse_args(["--pop-size", "200"])
        assert args.pop_size == 200

    def test_int_type(self):
        """2.2 Integer default → int type."""
        parser = create_parser({"count": 5})
        args = parser.parse_args(["--count", "10"])
        assert isinstance(args.count, int)
        assert args.count == 10

    def test_float_type(self):
        """2.2 Float default → float type."""
        parser = create_parser({"rate": 0.1})
        args = parser.parse_args(["--rate", "0.5"])
        assert isinstance(args.rate, float)
        assert args.rate == 0.5

    def test_string_type(self):
        """2.2 String default → str type."""
        parser = create_parser({"name": "hello"})
        args = parser.parse_args(["--name", "world"])
        assert isinstance(args.name, str)
        assert args.name == "world"

    def test_bool_false_default_store_true(self):
        """2.3 Boolean False default → store_true."""
        parser = create_parser({"use_cuda": False})
        # Without flag: default is absent (store_true default is False)
        args = parser.parse_args([])
        assert args.use_cuda is False
        # With flag
        args = parser.parse_args(["--use-cuda"])
        assert args.use_cuda is True

    def test_bool_true_default_store_false(self):
        """2.3 Boolean True default → store_false."""
        parser = create_parser({"verbose": True})
        args = parser.parse_args([])  # noqa: F841
        # store_false: without flag → default True (True becomes store_false action,
        # which means the flag negates it)
        # store_false sets the attribute to False when the flag is passed
        args_with_flag = parser.parse_args(["--verbose"])
        assert args_with_flag.verbose is False

    def test_empty_argv_returns_defaults(self):
        """2.4 Empty argv returns all defaults unchanged."""
        parser = create_parser(SAMPLE_PARAMS)
        args = parser.parse_args([])
        assert args.pop_size == 100
        assert args.learning_rate == 0.01
        assert args.algorithm_name == "adam"
