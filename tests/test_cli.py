"""CLI smoke tests for box command."""

import pytest
from click.testing import CliRunner

from boxing_gym.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


class TestMainGroup:
    """Test the main CLI group."""

    def test_help_works(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "BoxingGym benchmark analysis CLI" in result.output

    def test_version_works(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "box" in result.output


class TestSyncCommand:
    """Test sync command basics."""

    def test_sync_help(self, runner):
        result = runner.invoke(main, ["sync", "--help"])
        assert result.exit_code == 0
        assert "--local" in result.output
        assert "--status" in result.output

    def test_sync_status_runs(self, runner):
        result = runner.invoke(main, ["sync", "--status"])
        # exit 0 or 1 both acceptable (depends on cache state)
        assert result.exit_code in (0, 1)

    def test_sync_nonexistent_dir_fails(self, runner):
        result = runner.invoke(main, ["sync", "--local", "/nonexistent/path"])
        # click validates path exists, so this should fail
        assert result.exit_code != 0


class TestQueryCommand:
    """Test query command basics."""

    def test_query_help(self, runner):
        result = runner.invoke(main, ["query", "--help"])
        assert result.exit_code == 0
        assert "leaderboard" in result.output
        assert "--format" in result.output

    def test_query_list_works(self, runner):
        result = runner.invoke(main, ["query", "--list"])
        assert result.exit_code == 0
        assert "leaderboard" in result.output
        assert "oed-discovery" in result.output

    def test_query_unknown_name_fails(self, runner):
        result = runner.invoke(main, ["query", "not-a-real-query"])
        # should fail gracefully with unknown query
        assert (
            result.exit_code != 0
            or "not found" in result.output.lower()
            or "unknown" in result.output.lower()
        )


class TestResultsCommand:
    """Test results command basics."""

    def test_results_help(self, runner):
        result = runner.invoke(main, ["results", "--help"])
        assert result.exit_code == 0
        assert "--view" in result.output
        assert "--tui" in result.output
        assert "--web" in result.output

    def test_results_default_runs(self, runner):
        # default results without data may fail, but shouldn't crash
        result = runner.invoke(main, ["results"])
        # accept 0 (success) or 1 (no data) but not other codes
        assert result.exit_code in (0, 1)


class TestCommandIntegration:
    """Test commands work together."""

    def test_sync_then_query_pattern(self, runner, tmp_path):
        # sync to empty dir
        result = runner.invoke(main, ["sync", "--local", str(tmp_path)])
        # query should still list available queries
        result = runner.invoke(main, ["query", "--list"])
        assert result.exit_code == 0
        assert "leaderboard" in result.output
