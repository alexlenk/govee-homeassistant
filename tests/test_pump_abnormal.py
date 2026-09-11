"""Pump-fault status for pump-model dehumidifiers (H7152 "Max") — issue #114 follow-up.

Reverse-engineered from a live clogged-drain capture on a real H7152: neither
the OpenAPI event-push channel nor the flat MQTT ``state`` keys carried any
trace of the fault (confirmed empty across two captures, two minutes apart,
while the app showed "Pump Abnormality"). The only signal is byte offset 12
of an ``aa 17`` status frame riding in the AWS IoT push's ``op.command`` list
— ``0x00`` normally, ``0x01`` while the fault is active. The frames below are
taken verbatim from real diagnostics downloads (before / during / after an
actual fault).
"""

from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.govee.models import GoveeCapability, GoveeDevice, GoveeDeviceState
from custom_components.govee.models.device import CAPABILITY_ON_OFF, INSTANCE_POWER

DEVICE_ID = "11:66:C0:EB:1D:75:5C:E7"

# Verbatim ``aa 17`` frames from real diagnostics captures.
FRAME_PUMP_OK = bytes.fromhex("aa170000000000000000000000000000000000bd")
FRAME_PUMP_ABNORMAL = bytes.fromhex("aa170000000000000000000001000000000000bc")
FRAME_PUMP_RECOVERED = bytes.fromhex("aa170000000000000000000000000000000000bd")

# An unrelated frame from the same push, to prove the scan doesn't false-match.
FRAME_UNRELATED = bytes.fromhex("aa050003000000000000000000000000000000ac")

# Verbatim ``aa 10 81 03`` frames from 5 real (app-screenshot + diagnostics)
# capture pairs on 2026-09-10 — see update_temperature_from_frames for the
# exact packed-value decode these confirm (superseding an earlier best-effort
# linear fit). Named by their app-displayed temperature.
FRAME_TEMP_69_6F = bytes.fromhex("aa10810332ce00000000000000000000000000c4")
FRAME_TEMP_70_7F = bytes.fromhex("aa1081034a370000000000000000000000000045")
FRAME_TEMP_71_1F = bytes.fromhex("aa10810355ff0000000000000000000000000092")
FRAME_TEMP_72_1F = bytes.fromhex("aa10810369c50000000000000000000000000094")
FRAME_TEMP_72_3F = bytes.fromhex("aa1081036db000000000000000000000000000e5")

# Verbatim ``aa 19`` frames from a live hose-button test on 2026-09-11 —
# byte offset 9 confirmed against the app's own Mode label toggling in exact
# lockstep across five press-and-hold cycles. See
# update_dehumidifier_mode_from_frames for the full story, including why
# byte offset 7 (which also moves) is deliberately NOT decoded here.
FRAME_MODE_PUMP = bytes.fromhex("aa190000000000010101000000000000000000b2")
FRAME_MODE_TANK = bytes.fromhex("aa190000000000010100000000000000000000b3")


def _h7152() -> GoveeDevice:
    return GoveeDevice(
        device_id=DEVICE_ID,
        sku="H7152",
        name="Smart Dehumidifier Max",
        device_type="devices.types.dehumidifier",
        capabilities=(GoveeCapability(type=CAPABILITY_ON_OFF, instance=INSTANCE_POWER, parameters={}),),
    )


def _h7150() -> GoveeDevice:
    """Non-pump variant — must NOT be treated as pump-capable."""
    return GoveeDevice(
        device_id="11:66:C0:EB:1D:75:5C:E8",
        sku="H7150",
        name="Smart Dehumidifier",
        device_type="devices.types.dehumidifier",
        capabilities=(GoveeCapability(type=CAPABILITY_ON_OFF, instance=INSTANCE_POWER, parameters={}),),
    )


class TestSupportsPumpAbnormal:
    def test_h7152_supports_pump_abnormal(self):
        assert _h7152().supports_pump_abnormal is True

    def test_h7150_does_not_support_pump_abnormal(self):
        """H7150 is a non-pump variant — no confirmed frame layout for it yet."""
        assert _h7150().supports_pump_abnormal is False


class TestUpdatePumpAbnormalFromFrames:
    def test_normal_frame_clears_flag(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_pump_abnormal_from_frames([FRAME_PUMP_OK]) is True
        assert state.pump_abnormal is False

    def test_fault_frame_sets_flag(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_pump_abnormal_from_frames([FRAME_PUMP_ABNORMAL]) is True
        assert state.pump_abnormal is True

    def test_flag_clears_again_on_recovery(self):
        """Live/level flag, not edge-latched like water_full — it tracks live."""
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        state.update_pump_abnormal_from_frames([FRAME_PUMP_ABNORMAL])
        assert state.pump_abnormal is True

        state.update_pump_abnormal_from_frames([FRAME_PUMP_RECOVERED])
        assert state.pump_abnormal is False

    def test_unrelated_frame_is_not_recognised(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_pump_abnormal_from_frames([FRAME_UNRELATED]) is False
        assert state.pump_abnormal is None

    def test_short_frame_is_ignored(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        short = bytes([0xAA, 0x17, 0x00])
        assert state.update_pump_abnormal_from_frames([short]) is False
        assert state.pump_abnormal is None

    def test_picks_the_right_frame_out_of_a_full_push(self):
        """A real push carries ~15 frames; the scan must find the aa 17 one."""
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        frames = [FRAME_UNRELATED, FRAME_PUMP_ABNORMAL]
        assert state.update_pump_abnormal_from_frames(frames) is True
        assert state.pump_abnormal is True


class TestSupportsTemperatureSensorOnPumpDehumidifier:
    def test_h7152_supports_temperature_sensor(self):
        assert _h7152().supports_temperature_sensor is True

    def test_h7150_does_not_support_temperature_sensor(self):
        """No sensorTemperature capability and not confirmed on H7150 frames."""
        assert _h7150().supports_temperature_sensor is False


class TestUpdateTemperatureFromFrames:
    def test_recognises_the_frame(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_temperature_from_frames([FRAME_TEMP_70_7F]) is True

    def test_unrelated_frame_is_not_recognised(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_temperature_from_frames([FRAME_UNRELATED]) is False
        assert state.sensor_temperature is None

    def test_pump_frame_does_not_false_match(self):
        """Different 4-byte header (aa 17 vs aa 10 81 03) — must not collide."""
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_temperature_from_frames([FRAME_PUMP_OK]) is False

    @pytest.mark.parametrize(
        "frame,expected_celsius,expected_humidity",
        [
            (FRAME_TEMP_69_6F, 20.9, 61.4),
            (FRAME_TEMP_70_7F, 21.5, 60.7),
            (FRAME_TEMP_71_1F, 21.8, 62.3),
            (FRAME_TEMP_72_1F, 22.3, 68.5),
            (FRAME_TEMP_72_3F, 22.4, 68.8),
        ],
    )
    def test_decodes_the_exact_value(self, frame, expected_celsius, expected_humidity):
        """Bytes 3-5 are a single big-endian packed value — temperature and
        humidity each x10 and concatenated, matching the app's own
        ``CmdStatusParseV1``/``ThermometerInfo`` decode (see the method's
        docstring). This is an exact decode confirmed against 9 real capture
        pairs with zero residual error, not a fit — these 5 are the
        app-screenshot-confirmed temperature pairs; the humidity half of each
        is the same formula applied to the same verbatim frame."""
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        state.update_temperature_from_frames([frame])
        assert state.sensor_temperature == expected_celsius
        assert state.sensor_humidity == expected_humidity

    def test_picks_the_right_frame_out_of_a_full_push(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        frames = [FRAME_UNRELATED, FRAME_PUMP_OK, FRAME_TEMP_72_3F]
        assert state.update_temperature_from_frames(frames) is True
        assert state.sensor_temperature == 22.4
        assert state.sensor_humidity == 68.8


class TestSupportsHumiditySensorOnPumpDehumidifier:
    def test_h7152_supports_humidity_sensor(self):
        assert _h7152().supports_humidity_sensor is True

    def test_h7150_does_not_support_humidity_sensor(self):
        """No sensorHumidity capability and not confirmed on H7150 frames."""
        assert _h7150().supports_humidity_sensor is False


class TestUpdateDehumidifierModeFromFrames:
    """Byte offset 9 of the ``aa 19`` frame, confirmed against the app's own
    Mode label toggling in lockstep across a live hose-button test — see
    FRAME_MODE_PUMP/FRAME_MODE_TANK's module-level comment.
    """

    def test_pump_frame_reports_pump(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_dehumidifier_mode_from_frames([FRAME_MODE_PUMP]) is True
        assert state.dehumidifier_mode == "pump"

    def test_tank_frame_reports_tank(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_dehumidifier_mode_from_frames([FRAME_MODE_TANK]) is True
        assert state.dehumidifier_mode == "tank"

    def test_toggles_both_ways(self):
        """Live/level flag, not edge-latched — tracks the current mode."""
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        state.update_dehumidifier_mode_from_frames([FRAME_MODE_TANK])
        assert state.dehumidifier_mode == "tank"

        state.update_dehumidifier_mode_from_frames([FRAME_MODE_PUMP])
        assert state.dehumidifier_mode == "pump"

    def test_unrelated_frame_is_not_recognised(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        assert state.update_dehumidifier_mode_from_frames([FRAME_UNRELATED]) is False
        assert state.dehumidifier_mode is None

    def test_short_frame_is_ignored(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        short = bytes([0xAA, 0x19, 0x00])
        assert state.update_dehumidifier_mode_from_frames([short]) is False
        assert state.dehumidifier_mode is None

    def test_picks_the_right_frame_out_of_a_full_push(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        frames = [FRAME_UNRELATED, FRAME_PUMP_OK, FRAME_MODE_TANK]
        assert state.update_dehumidifier_mode_from_frames(frames) is True
        assert state.dehumidifier_mode == "tank"


class TestPumpAbnormalPreservedAcrossDeveloperPoll:
    """The Developer /device/state poll has no field for this at all — it
    only ever comes from AWS IoT push frames — so a naive poll would flicker
    the sensor to "unknown" every ~60s (same bug class as water_full/presence,
    issues #118/#124; caught live on a real device where the entity flapped
    OK/Unknown every poll cycle after the initial implementation).
    """

    def _coord(self):
        import custom_components.govee.coordinator as coord_mod

        hass = MagicMock()
        config_entry = MagicMock()
        config_entry.entry_id = "test_entry"
        config_entry.async_create_background_task = MagicMock()
        coord = coord_mod.GoveeCoordinator(
            hass=hass,
            config_entry=config_entry,
            api_client=MagicMock(),
            iot_credentials=None,
            poll_interval=60,
        )
        coord._devices[DEVICE_ID] = GoveeDevice(
            device_id=DEVICE_ID,
            sku="H7152",
            name="Smart Dehumidifier Max",
            device_type="devices.types.dehumidifier",
            capabilities=(GoveeCapability(type=CAPABILITY_ON_OFF, instance=INSTANCE_POWER, parameters={}),),
            is_group=False,
        )
        return coord

    @pytest.mark.asyncio
    async def test_fault_survives_a_poll_that_knows_nothing_about_it(self):
        coord = self._coord()
        existing = GoveeDeviceState.create_empty(DEVICE_ID)
        existing.pump_abnormal = True
        coord._states[DEVICE_ID] = existing

        # What the Developer poll actually returns: no pump_abnormal field at
        # all, so the fresh state has it as None.
        fresh = GoveeDeviceState.create_empty(DEVICE_ID)
        coord._api_client.get_device_state = AsyncMock(return_value=fresh)

        result = await coord._fetch_device_state(DEVICE_ID, coord._devices[DEVICE_ID])

        assert result.pump_abnormal is True

    @pytest.mark.asyncio
    async def test_dehumidifier_mode_survives_a_poll_that_knows_nothing_about_it(self):
        coord = self._coord()
        existing = GoveeDeviceState.create_empty(DEVICE_ID)
        existing.dehumidifier_mode = "tank"
        coord._states[DEVICE_ID] = existing

        fresh = GoveeDeviceState.create_empty(DEVICE_ID)
        coord._api_client.get_device_state = AsyncMock(return_value=fresh)

        result = await coord._fetch_device_state(DEVICE_ID, coord._devices[DEVICE_ID])

        assert result.dehumidifier_mode == "tank"

    @pytest.mark.asyncio
    async def test_recovery_is_not_masked_by_a_stale_preserved_value(self):
        """Preservation only fills a None gap — it must never overwrite a
        push-derived value the poll legitimately doesn't touch."""
        coord = self._coord()
        existing = GoveeDeviceState.create_empty(DEVICE_ID)
        existing.pump_abnormal = False
        coord._states[DEVICE_ID] = existing

        fresh = GoveeDeviceState.create_empty(DEVICE_ID)
        coord._api_client.get_device_state = AsyncMock(return_value=fresh)

        result = await coord._fetch_device_state(DEVICE_ID, coord._devices[DEVICE_ID])

        assert result.pump_abnormal is False


class TestDehumidifierModeSensor:
    def _coord(self):
        import custom_components.govee.coordinator as coord_mod

        hass = MagicMock()
        config_entry = MagicMock()
        config_entry.entry_id = "test_entry"
        coord = coord_mod.GoveeCoordinator(
            hass=hass,
            config_entry=config_entry,
            api_client=MagicMock(),
            iot_credentials=None,
            poll_interval=60,
        )
        coord._devices[DEVICE_ID] = _h7152()
        return coord

    def test_native_value_reflects_state(self):
        from custom_components.govee.sensor import GoveeDehumidifierModeSensor

        coord = self._coord()
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        state.dehumidifier_mode = "tank"
        coord._states[DEVICE_ID] = state

        sensor = GoveeDehumidifierModeSensor(coord, coord._devices[DEVICE_ID])
        assert sensor.native_value == "tank"

    def test_native_value_none_before_any_push(self):
        from custom_components.govee.sensor import GoveeDehumidifierModeSensor

        coord = self._coord()
        sensor = GoveeDehumidifierModeSensor(coord, coord._devices[DEVICE_ID])
        assert sensor.native_value is None
