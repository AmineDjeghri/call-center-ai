from datetime import tzinfo
from pydantic_extra_types.phone_numbers import PhoneNumber as PydanticPhoneNumber
from pytz import country_timezones, timezone, utc
import phonenumbers


class PhoneNumber(PydanticPhoneNumber):
    parsed: phonenumbers.PhoneNumber
    phone_format = "E164"  # E164 is standard accross all Microsoft services

    def __init__(self) -> None:
        self.parsed = phonenumbers.parse(self)

    def tz(self) -> tzinfo:
        """
        Return timezone of a phone number.

        If the country code cannot be determined, return UTC as a fallback.
        """
        if not self.parsed.country_code:
            return utc
        region_code = phonenumbers.region_code_for_country_code(
            self.parsed.country_code
        )
        tz_name = country_timezones[region_code][0]
        return timezone(tz_name)

    def __eq__(self, other: object) -> bool:
        return self.parsed == other.parsed if isinstance(other, PhoneNumber) else False
