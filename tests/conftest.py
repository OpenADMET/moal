"""Pytest configuration and shared fixtures for the moal test suite.

This module is loaded by pytest before any test module is imported.
Setting the matplotlib backend to "Agg" here — before pyplot is first
imported — ensures that no interactive window is ever created during the
test run, regardless of the ``show=`` argument passed to ``LiveDashboard``.
"""

import matplotlib

matplotlib.use("Agg")
