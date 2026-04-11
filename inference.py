"""
Inference Script — Autonomous Traffic Control OpenEnv Environment
=================================================================
Advanced Hybrid Agent: combines optimized rule engine + LLM for ambiguous cases.

Mandatory env variables (injected by validator):
    API_BASE_URL   LLM proxy endpoint (MUST use validator's LiteLLM proxy)
    API_KEY        LiteLLM proxy key
    MODEL_NAME     Model identifier

Optional:
    SERVER_URL     Running env server (default: http://localhost:8000)

Output format: [START], [STEP], [END] lines only (strict protocol compliance)
"""

import os
import re
import sys
import json
import textwrap
import requests as _http
from typing import List, Optional

# Allow running from repo root or from traffic_control/ subdirectory
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

try:
    from traffic_control.client import TrafficControlEnv
    from traffic_control.models import TrafficAction, TrafficObservation
except ImportError:
    from client import TrafficControlEnv  # type: ignore
    from models import TrafficAction, TrafficObservation  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CRITICAL: Must use os.environ[] with NO fallbacks per validator requirements
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

SEED        = 42
MAX_TOKENS  = 64
TEMPERATURE = 0.0
