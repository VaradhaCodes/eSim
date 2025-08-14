"""
Plotting Utilities Module

This module provides utility functions for running the plotting application
as a standalone program with command line arguments.
"""

import sys
import logging
from typing import List
from PyQt5.QtWidgets import QApplication
from .plot_window import plotWindow

# Set up logging
logger = logging.getLogger(__name__)

# Default values
DEFAULT_FILE_PATH = "."
DEFAULT_PROJECT_NAME = "Test Project"


def parse_command_line_arguments(args: List[str]) -> tuple[str, str]:
    """
    Parse command line arguments for file path and project name.
    
    Args:
        args: Command line arguments list
        
    Returns:
        Tuple containing (file_path, project_name)
    """
    if len(args) > 2:
        file_path = args[1]
        project_name = args[2]
        logger.info(f"Using command line arguments: path={file_path}, project={project_name}")
    else:
        file_path = DEFAULT_FILE_PATH
        project_name = DEFAULT_PROJECT_NAME
        logger.info(f"Using default values: path={file_path}, project={project_name}")
    
    return file_path, project_name


def main() -> None:
    """
    Main entry point for the plotting application.
    
    Creates a QApplication, parses command line arguments, creates the plot window,
    and starts the application event loop.
    """
    try:
        # Create the application
        app = QApplication(sys.argv)
        app.setApplicationName("Python Plotting")
        app.setApplicationVersion("1.0.0")
        
        # Parse command line arguments
        file_path, project_name = parse_command_line_arguments(sys.argv)
        
        # Create and show the main window
        main_window = plotWindow(file_path, project_name)
        main_window.show()
        
        logger.info("Application started successfully")
        
        # Start the event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
