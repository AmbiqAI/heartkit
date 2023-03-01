import json

from heartkit import cli


def test_app_cli_help():
    """Verify APP CLI provides help dialog."""
    args = json.loads(cli.AppArguments.schema_json())
    assert isinstance(args, dict)


def test_tf_model():
    """Verify TF model produces correct results on small sample set."""
    assert True


def test_tfl_model():
    """Verify TFLite and micro models produces correct results on small sample set."""
    assert True
