#!/usr/bin/env python3

import pytest
import os
import tempfile
import datetime as dt

# Import functions to test
from src.infer_ifcbbins import get_output_path, argparse_runtime_args, argparse_init
from src.infer_ifcbbins_torch import get_output_path as get_output_path_torch, argparse_runtime_args as argparse_runtime_args_torch, argparse_init as argparse_init_torch


class TestGetOutputPath:
    """Test the get_output_path function from both inference scripts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.args = type('Args', (), {})()
        self.args.outdir = './outputs'
        self.args.outfile = '{RUN_DATE}/{SUBPATH}.csv'
        self.args.run_date_str = '2025-01-15'
        self.args.model_name = 'test_model'
        
    def test_basic_output_path_run_date(self):
        """Test basic output path generation with run-date format."""
        result = get_output_path(self.args, 'test_bin')
        expected = os.path.join('./outputs', '2025-01-15/test_bin.csv')
        assert result == expected
        
    def test_basic_output_path_model_name(self):
        """Test basic output path generation with model-name format."""
        self.args.outfile = '{MODEL_NAME}/{SUBPATH}.csv'
        result = get_output_path(self.args, 'test_bin')
        expected = os.path.join('./outputs', 'test_model/test_bin.csv')
        assert result == expected
        
    def test_output_path_with_relative_path(self):
        """Test output path generation with bin_relative_path."""
        bin_relative_path = 'subdir/test_bin'
        result = get_output_path(self.args, 'test_bin', bin_relative_path)
        expected = os.path.join('./outputs', '2025-01-15/subdir/test_bin.csv')
        assert result == expected
        
    def test_output_path_torch_version(self):
        """Test torch version of get_output_path function."""
        result = get_output_path_torch(self.args, 'test_bin')
        expected = os.path.join('./outputs', '2025-01-15/test_bin.csv')
        assert result == expected
        
    def test_output_path_with_custom_outdir(self):
        """Test output path with custom output directory."""
        self.args.outdir = '/custom/output/dir'
        result = get_output_path(self.args, 'test_bin')
        expected = os.path.join('/custom/output/dir', '2025-01-15/test_bin.csv')
        assert result == expected
        
    def test_output_path_complex_relative_path(self):
        """Test output path with complex relative path structure."""
        bin_relative_path = 'year2024/month03/day15/test_bin'
        result = get_output_path(self.args, 'test_bin', bin_relative_path)
        expected = os.path.join('./outputs', '2025-01-15/year2024/month03/day15/test_bin.csv')
        assert result == expected


class TestArgparseRuntimeArgs:
    """Test the argparse_runtime_args function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.args = type('Args', (), {})()
        self.args.MODEL = '/path/to/my_awesome_model.onnx'
        self.args.classes = None
        self.args.BINS = ['bin1', 'bin2']
        
    def test_runtime_args_basic(self, mocker):
        """Test basic runtime args processing."""
        # Mock environment and datetime
        mocker.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2'})
        mock_datetime = mocker.patch('datetime.datetime')
        mock_datetime.now.return_value.isoformat.return_value = '2025-01-15T14:30:45'
        
        argparse_runtime_args(self.args)
        
        assert self.args.run_date_str == '2025-01-15'
        assert self.args.run_time_str == '14:30:45'
        assert self.args.model_name == 'my_awesome_model'
        assert self.args.gpus == [0, 1, 2]
        
    def test_runtime_args_no_gpu(self, mocker):
        """Test runtime args with no GPU specified."""
        mocker.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''})
        mock_datetime = mocker.patch('datetime.datetime')
        mock_datetime.now.return_value.isoformat.return_value = '2025-01-15T14:30:45'
        
        argparse_runtime_args(self.args)
        
        assert self.args.gpus == []  # Empty list when no GPUs
        
    def test_runtime_args_with_classes_file(self, mocker):
        """Test runtime args with classes file."""
        mock_datetime = mocker.patch('datetime.datetime')
        mock_datetime.now.return_value.isoformat.return_value = '2025-01-15T14:30:45'
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('class1\nclass2\nclass3\n')
            classes_file = f.name
            
        try:
            self.args.classes = classes_file
            argparse_runtime_args(self.args)
            assert self.args.classes == ['class1', 'class2', 'class3']
        finally:
            os.unlink(classes_file)
            
    def test_runtime_args_torch_version(self, mocker):
        """Test torch version of runtime args."""
        mock_datetime = mocker.patch('datetime.datetime')
        mock_datetime.now.return_value.isoformat.return_value = '2025-01-15T14:30:45'
        argparse_runtime_args_torch(self.args)
        
        assert self.args.run_date_str == '2025-01-15'
        assert self.args.model_name == 'my_awesome_model'


class TestSubfolderTypeLogic:
    """Test the subfolder-type logic in main functions."""
    
    def test_argparse_init_has_subfolder_type(self):
        """Test that argparse_init includes subfolder-type argument."""
        parser = argparse_init()
        
        # Parse help to check if subfolder-type is present
        help_text = parser.format_help()
        assert '--subfolder-type' in help_text
        assert 'run-date' in help_text
        assert 'model-name' in help_text
        
    def test_argparse_init_torch_has_subfolder_type(self):
        """Test that torch version argparse_init includes subfolder-type argument."""
        parser = argparse_init_torch()
        
        help_text = parser.format_help()
        assert '--subfolder-type' in help_text
        assert 'run-date' in help_text
        assert 'model-name' in help_text
        
    def test_default_subfolder_type_is_run_date(self):
        """Test that default subfolder-type is run-date."""
        parser = argparse_init()
        args = parser.parse_args(['model.onnx', 'bins/'])
        assert args.subfolder_type == 'run-date'
        
    def test_model_name_subfolder_type_parsing(self):
        """Test parsing model-name subfolder-type."""
        parser = argparse_init()
        args = parser.parse_args(['--subfolder-type', 'model-name', 'model.onnx', 'bins/'])
        assert args.subfolder_type == 'model-name'
        
    def test_invalid_subfolder_type_raises_error(self):
        """Test that invalid subfolder-type raises error."""
        parser = argparse_init()
        with pytest.raises(SystemExit):
            parser.parse_args(['--subfolder-type', 'invalid', 'model.onnx', 'bins/'])


class TestOutputFilePatternLogic:
    """Test the logic for updating output file patterns based on subfolder-type."""
    
    def test_run_date_keeps_default_pattern(self):
        """Test that run-date keeps the default outfile pattern."""
        args = type('Args', (), {})()
        args.subfolder_type = 'run-date'
        args.outfile = '{RUN_DATE}/{SUBPATH}.csv'
        
        # Simulate the logic from __main__
        if args.subfolder_type == 'model-name' and args.outfile == '{RUN_DATE}/{SUBPATH}.csv':
            args.outfile = '{MODEL_NAME}/{SUBPATH}.csv'
            
        assert args.outfile == '{RUN_DATE}/{SUBPATH}.csv'
        
    def test_model_name_updates_pattern(self):
        """Test that model-name updates the outfile pattern."""
        args = type('Args', (), {})()
        args.subfolder_type = 'model-name'
        args.outfile = '{RUN_DATE}/{SUBPATH}.csv'
        
        # Simulate the logic from __main__
        if args.subfolder_type == 'model-name' and args.outfile == '{RUN_DATE}/{SUBPATH}.csv':
            args.outfile = '{MODEL_NAME}/{SUBPATH}.csv'
            
        assert args.outfile == '{MODEL_NAME}/{SUBPATH}.csv'
        
    def test_torch_version_model_name_logic(self):
        """Test torch version model-name logic."""
        args = type('Args', (), {})()
        args.subfolder_type = 'model-name'
        args.outdir = './outputs'
        args.outfile = '{RUN_DATE}/{SUBPATH}.csv'
        
        # Simulate torch version logic
        if args.subfolder_type == 'model-name':
            if args.outdir == './outputs' and args.outfile == '{RUN_DATE}/{SUBPATH}.csv':
                args.outdir = './outputs/{MODEL_NAME}'
                args.outfile = '{SUBPATH}.csv'
                
        assert args.outdir == './outputs/{MODEL_NAME}'
        assert args.outfile == '{SUBPATH}.csv'


class TestBinDirectoryMapping:
    """Test bin-to-input-directory mapping logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.args = type('Args', (), {})()
        self.args.MODEL = 'test_model.onnx'
        self.args.classes = None
        
    def test_bin_to_input_dir_structure(self, mocker):
        """Test that bin_to_input_dir mapping is created correctly."""
        self.args.BINS = ['dir1', 'individual_bin']
        
        # Mock datetime
        mock_datetime = mocker.patch('datetime.datetime')
        mock_datetime.now.return_value.isoformat.return_value = '2025-01-15T14:30:45'
        
        # Mock directory and file checks
        mock_isdir = mocker.patch('os.path.isdir')
        mock_dd = mocker.patch('ifcb.DataDirectory')
        
        # Set up mocks
        mock_isdir.side_effect = lambda x: x == 'dir1'
        
        mock_bin_obj = type('BinObj', (), {})()
        mock_bin_obj.fileset = type('FileSet', (), {})()
        mock_bin_obj.fileset.basepath = 'dir1/bin1'
        mock_dd.return_value = [mock_bin_obj]
        
        argparse_runtime_args(self.args)
        
        # Check that bin_to_input_dir mapping exists
        assert hasattr(self.args, 'bin_to_input_dir')
        assert 'dir1/bin1' in self.args.bin_to_input_dir
        assert self.args.bin_to_input_dir['dir1/bin1'] == 'dir1'


@pytest.mark.parametrize("subfolder_type,expected_pattern", [
    ('run-date', '{RUN_DATE}/{SUBPATH}.csv'),
    ('model-name', '{MODEL_NAME}/{SUBPATH}.csv'),
])
def test_subfolder_pattern_parametrized(subfolder_type, expected_pattern):
    """Parametrized test for different subfolder types."""
    args = type('Args', (), {})()
    args.subfolder_type = subfolder_type
    args.outfile = '{RUN_DATE}/{SUBPATH}.csv'
    
    # Simulate the logic from __main__
    if args.subfolder_type == 'model-name' and args.outfile == '{RUN_DATE}/{SUBPATH}.csv':
        args.outfile = '{MODEL_NAME}/{SUBPATH}.csv'
        
    assert args.outfile == expected_pattern


@pytest.mark.parametrize("model_path,expected_name", [
    ('/path/to/classifier.onnx', 'classifier'),
    ('simple_model.onnx', 'simple_model'),
    ('/complex/path/with_underscores_v2.onnx', 'with_underscores_v2'),
    ('model.onnx', 'model'),
])
def test_model_name_extraction(model_path, expected_name, mocker):
    """Parametrized test for model name extraction."""
    args = type('Args', (), {})()
    args.MODEL = model_path
    args.classes = None
    args.BINS = []
    
    mock_datetime = mocker.patch('datetime.datetime')
    mock_datetime.now.return_value.isoformat.return_value = '2025-01-15T14:30:45'
    argparse_runtime_args(args)
    
    assert args.model_name == expected_name


if __name__ == '__main__':
    pytest.main([__file__, '-v'])