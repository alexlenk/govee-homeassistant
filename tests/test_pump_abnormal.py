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
import json
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
# regression this was fit from. Named by their app-displayed temperature.
FRAME_TEMP_69_6F = bytes.fromhex("aa10810332ce00000000000000000000000000c4")
FRAME_TEMP_70_7F = bytes.fromhex("aa1081034a370000000000000000000000000045")
FRAME_TEMP_71_1F = bytes.fromhex("aa10810355ff0000000000000000000000000092")
FRAME_TEMP_72_1F = bytes.fromhex("aa10810369c50000000000000000000000000094")
FRAME_TEMP_72_3F = bytes.fromhex("aa1081036db000000000000000000000000000e5")


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
        "frame,expected_celsius",
        [
            (FRAME_TEMP_69_6F, 20.9),
            (FRAME_TEMP_70_7F, 21.5),
            (FRAME_TEMP_71_1F, 21.8),
            (FRAME_TEMP_72_1F, 22.3),
            (FRAME_TEMP_72_3F, 22.4),
        ],
    )
    def test_decodes_the_fitted_value(self, frame, expected_celsius):
        """Applies the regression constants to the captured byte — the
        constants themselves are a best-effort fit (see the method's
        docstring for the residuals against the real app readings), this
        just locks in that the arithmetic on a known input doesn't drift."""
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        state.update_temperature_from_frames([frame])
        assert state.sensor_temperature == expected_celsius

    def test_picks_the_right_frame_out_of_a_full_push(self):
        state = GoveeDeviceState.create_empty(DEVICE_ID)
        frames = [FRAME_UNRELATED, FRAME_PUMP_OK, FRAME_TEMP_72_3F]
        assert state.update_temperature_from_frames(frames) is True
        assert state.sensor_temperature == 22.4


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


class TestH7152DebugLog:
    """TEMPORARY debug-log capture (see coordinator._append_h7152_debug_line).

    Not a permanent feature — remove alongside it once byte 5 (humidity) and
    the tank-vs-pump-mode signal are both identified.
    """

    def _coord(self, tmp_path):
        import custom_components.govee.coordinator as coord_mod

        hass = MagicMock()
        log_path = tmp_path / "govee_h7152_debug.jsonl"
        hass.config.path.return_value = str(log_path)

        async def _run_in_executor(fn, *args):
            return fn(*args)

        hass.async_add_executor_job = AsyncMock(side_effect=_run_in_executor)

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
        return coord, log_path

    @staticmethod
    def _lines(log_path):
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text().splitlines()]

    @pytest.mark.asyncio
    async def test_mqtt_push_logs_every_frame_not_just_known_ones(self, tmp_path):
        """The whole point is catching signals we haven't decoded yet — a
        pre-filtered log defeats that."""
        coord, log_path = self._coord(tmp_path)
        state_data = {
            "onOff": 1,
            "sta": {"stc": "19_0_38_43170_1"},
            "result": 1,
            "_op_frames": [
                FRAME_UNRELATED.hex(),
                FRAME_TEMP_70_7F.hex(),
                FRAME_PUMP_OK.hex(),
            ],
        }

        await coord._log_h7152_debug_frame(state_data, sensor_temperature=21.5, pump_abnormal=False)

        lines = self._lines(log_path)
        assert len(lines) == 1
        assert lines[0]["source"] == "mqtt"
        assert lines[0]["op_frames_hex"] == [
            FRAME_UNRELATED.hex(),
            FRAME_TEMP_70_7F.hex(),
            FRAME_PUMP_OK.hex(),
        ]
        assert lines[0]["sta"] == {"stc": "19_0_38_43170_1"}
        assert lines[0]["sensor_temperature_c"] == 21.5
        assert lines[0]["pump_abnormal"] is False

    @pytest.mark.asyncio
    async def test_openapi_event_is_logged_too(self, tmp_path):
        coord, log_path = self._coord(tmp_path)

        await coord._log_h7152_debug_openapi_event("waterFullEvent", [{"name": "waterFull", "value": 1}])

        lines = self._lines(log_path)
        assert len(lines) == 1
        assert lines[0]["source"] == "openapi_event"
        assert lines[0]["instance"] == "waterFullEvent"
        assert lines[0]["state"] == [{"name": "waterFull", "value": 1}]

    @pytest.mark.asyncio
    async def test_stops_appending_past_the_size_cap(self, tmp_path, monkeypatch):
        import custom_components.govee.coordinator as coord_mod

        monkeypatch.setattr(coord_mod, "_H7152_DEBUG_LOG_MAX_BYTES", 10)
        coord, log_path = self._coord(tmp_path)
        log_path.write_text("x" * 20)  # already past the (patched) 10-byte cap

        await coord._append_h7152_debug_line({"source": "mqtt", "ts": "now"})

        assert log_path.read_text() == "x" * 20  # untouched — nothing appended

    @pytest.mark.asyncio
    async def test_write_failure_does_not_raise(self, tmp_path):
        """A debug aid must never break real state handling."""
        coord, _log_path = self._coord(tmp_path)
        coord.hass.config.path.return_value = str(tmp_path / "no" / "such" / "dir" / "x.jsonl")

        await coord._append_h7152_debug_line({"source": "mqtt", "ts": "now"})  # must not raise
