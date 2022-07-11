import logging
import datetime

from garminconnect import (
    Garmin,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
    GarminConnectAuthenticationError,
)

# Configure debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Example dates
today = datetime.date.today()
lastweek = today - datetime.timedelta(days=7)

# API

# Initialize Garmin api with your credentials
api = Garmin("lucas.j.menger@gmail.com", "G4rm1nT3st")

# Login to Garmin Connect portal
api.login()

# USER INFO

# Get full name from profile
logger.info(api.get_full_name())

# Get unit system from profile
logger.info(api.get_unit_system())
