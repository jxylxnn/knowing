"""Tests for InteractiveCLI class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import io
import sys


class TestInteractiveCLIImport:
    """Tests for InteractiveCLI import and basic functionality."""
    
    def test_interactive_cli_import(self):
        """Test InteractiveCLI can be imported."""
        from src.query.interactive_cli import InteractiveCLI
        assert InteractiveCLI is not None
    
    def test_interactive_cli_has_prompt_method(self):
        """Test InteractiveCLI has _prompt_live_simulation method."""
        from src.query.interactive_cli import InteractiveCLI
        assert hasattr(InteractiveCLI, '_prompt_live_simulation')


class TestPromptLiveSimulation:
    """Tests for _prompt_live_simulation method (exception handling fix)."""
    
    @pytest.fixture
    def mock_cli(self):
        """Create a mock InteractiveCLI for testing."""
        from src.query.interactive_cli import InteractiveCLI
        
        with patch('src.query.interactive_cli.ProjectionLoader'), \
             patch('src.query.interactive_cli.QueryParser'):
            cli = InteractiveCLI.__new__(InteractiveCLI)
            cli._simulator = None
            cli._projections_cache = None
        return cli
    
    def test_prompt_returns_true_for_yes(self, mock_cli, monkeypatch):
        """Test that 'yes' input returns True."""
        monkeypatch.setattr('sys.stdin', io.StringIO('yes\n'))
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is True
    
    def test_prompt_returns_true_for_y(self, mock_cli, monkeypatch):
        """Test that 'y' input returns True."""
        monkeypatch.setattr('sys.stdin', io.StringIO('y\n'))
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is True
    
    def test_prompt_returns_false_for_no(self, mock_cli, monkeypatch):
        """Test that 'no' input returns False."""
        monkeypatch.setattr('sys.stdin', io.StringIO('no\n'))
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is False
    
    def test_prompt_returns_false_for_empty(self, mock_cli, monkeypatch):
        """Test that empty input returns False."""
        monkeypatch.setattr('sys.stdin', io.StringIO('\n'))
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is False
    
    def test_prompt_returns_false_for_random_input(self, mock_cli, monkeypatch):
        """Test that random input returns False."""
        monkeypatch.setattr('sys.stdin', io.StringIO('maybe\n'))
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is False
    
    def test_prompt_handles_eof_error(self, mock_cli, monkeypatch):
        """Test that EOFError is handled gracefully."""
        def raise_eof():
            raise EOFError()
        
        monkeypatch.setattr('builtins.input', raise_eof)
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is False
    
    def test_prompt_handles_os_error(self, mock_cli, monkeypatch):
        """Test that OSError is handled gracefully."""
        def raise_os_error():
            raise OSError("Input error")
        
        monkeypatch.setattr('builtins.input', raise_os_error)
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is False
    
    def test_prompt_does_not_catch_keyboard_interrupt(self, mock_cli, monkeypatch):
        """Test that KeyboardInterrupt is NOT caught (should propagate)."""
        def raise_keyboard_interrupt():
            raise KeyboardInterrupt()
        
        monkeypatch.setattr('builtins.input', raise_keyboard_interrupt)
        
        with pytest.raises(KeyboardInterrupt):
            mock_cli._prompt_live_simulation('Test Player')
    
    def test_prompt_does_not_catch_system_exit(self, mock_cli, monkeypatch):
        """Test that SystemExit is NOT caught (should propagate)."""
        def raise_system_exit():
            raise SystemExit()
        
        monkeypatch.setattr('builtins.input', raise_system_exit)
        
        with pytest.raises(SystemExit):
            mock_cli._prompt_live_simulation('Test Player')
    
    def test_prompt_case_insensitive(self, mock_cli, monkeypatch):
        """Test that input is case-insensitive."""
        monkeypatch.setattr('sys.stdin', io.StringIO('YES\n'))
        
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is True
        
        monkeypatch.setattr('sys.stdin', io.StringIO('Y\n'))
        result = mock_cli._prompt_live_simulation('Test Player')
        assert result is True


class TestInteractiveCLIStructure:
    """Tests for InteractiveCLI class structure."""
    
    def test_class_methods_exist(self):
        """Test that expected methods exist on InteractiveCLI."""
        from src.query.interactive_cli import InteractiveCLI
        
        expected_methods = [
            '_prompt_live_simulation',
            '_run_live_simulation',
            'run'
        ]
        
        for method in expected_methods:
            assert hasattr(InteractiveCLI, method), f"Missing method: {method}"
    
    def test_no_bare_except_in_class(self):
        """Test that there are no bare except clauses in the class."""
        import inspect
        from src.query.interactive_cli import InteractiveCLI
        
        source = inspect.getsource(InteractiveCLI)
        
        bare_except_patterns = ['except:', 'except :']
        for pattern in bare_except_patterns:
            assert pattern not in source, f"Found bare except clause: '{pattern}'"
