"""
server/app.py — PersonalFinanceEnv HTTP server using openenv_core.
Uses create_app() with a factory callable as required by the OpenEnv spec.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv_core.env_server import create_app
except ImportError:
    from openenv_core import create_app  # fallback

try:
    from server.finance_environment import PersonalFinanceEnvironment
    from server.models import FinanceAction, FinanceObservation
except ImportError:
    from finance_environment import PersonalFinanceEnvironment
    from models import FinanceAction, FinanceObservation

app = create_app(
    PersonalFinanceEnvironment,   # pass class, not instance
    FinanceAction,
    FinanceObservation,
    env_name="personal-finance-manager",
)
