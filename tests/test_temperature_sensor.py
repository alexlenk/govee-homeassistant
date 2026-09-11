"""Live temperature reading for pump-model dehumidifiers (H7152 "Max") —
issue #114 follow-up.

The H7152 has no ``sensorTemperature`` capability at all (confirmed: absent
from the discovered capabilities list even though the app shows a live
reading) — its temperature readout travels over the AWS IoT status push's
BLE-format ``op.command`` frames instead. Byte offset 4 of the
``aa 10 81 03`` frame correlates near-linearly with the app's displayed
temperature: least-squares fit over 5 real (app-screenshot + diagnostics)
capture pairs, spanning 20.9-22.4°C, residuals 2-4 raw counts.

The frames below are taken verbatim from those real capture pairs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.govee.coordinator import GoveeCoordinator
from custom_components.govee.models import GoveeCapability, GoveeDevice, GoveeDeviceState
from custom_components.govee.models.device import CAPABILITY_ON_OFF, INSTANCE_POWER
from custom_components.govee.transport_health import TransportHealthTracker

DEVICE_ID = "11:66:C0:EB:1D:75:5C:E7"

# An unrelated frame from the same push, to prove the scan doesn't false-match.
FRAME_UNRELATED = bytes.fromhex("aa050003000000000000000000000000000000ac")
FRAME_PUMP_OK = bytes.fromhex("aa170000000000000000000000000000000000bd")

# Verbatim ``aa 10 81 03`` frames from 5 real (app-screenshot + diagnostics)
# capture pairs. Named by their app-displayed temperature.
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
    """Non-pump variant — must NOT get the frame-based temperature reading."""
    return GoveeDevice(
        device_id="11:66:C0:EB:1D:75:5C:E8",
        sku="H7150",
        name="Smart Dehumidifier",
        device_type="devices.types.dehumidifier",
        capabilities=(GoveeCapability(type=CAPABILITY_ON_OFF, instance=INSTANCE_POWER, parameters={}),),
    )


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


class TestCoordinatorAppliesTemperatureFromAnMqttPush:
    """_on_mqtt_state_update must actually call the decoder for an H7152
    push — the decoder itself is covered above, this exercises the wiring
    in the coordinator that calls it."""

    def _coordinator(self) -> GoveeCoordinator:
        coordinator = GoveeCoordinator.__new__(GoveeCoordinator)
        coordinator._devices = {DEVICE_ID: _h7152()}
        coordinator._states = {DEVICE_ID: GoveeDeviceState.create_empty(DEVICE_ID)}
        coordinator._transport = TransportHealthTracker()
        coordinator.async_set_updated_data = MagicMock()
        return coordinator

    def test_temperature_applied_from_push(self):
        coordinator = self._coordinator()

        coordinator._on_mqtt_state_update(
            DEVICE_ID,
            {"onOff": 1, "_op_frames": [FRAME_TEMP_72_3F.hex()]},
        )

        state = coordinator._states[DEVICE_ID]
        assert state.sensor_temperature == 22.4

    def test_non_pump_device_is_left_untouched(self):
        coordinator = GoveeCoordinator.__new__(GoveeCoordinator)
        other_id = _h7150().device_id
        coordinator._devices = {other_id: _h7150()}
        coordinator._states = {other_id: GoveeDeviceState.create_empty(other_id)}
        coordinator._transport = TransportHealthTracker()
        coordinator.async_set_updated_data = MagicMock()

        coordinator._on_mqtt_state_update(
            other_id,
            {"onOff": 1, "_op_frames": [FRAME_TEMP_72_3F.hex()]},
        )

        assert coordinator._states[other_id].sensor_temperature is None


class TestSensorSetupRegistersTemperatureForH7152:
    """async_setup_entry must actually create the entity for an H7152 via
    the existing generic GoveeTemperatureSensor — exercises the widened
    supports_temperature_sensor gate specifically."""

    @pytest.mark.asyncio
    async def test_h7152_gets_a_temperature_entity(self):
        from custom_components.govee.sensor import GoveeTemperatureSensor, async_setup_entry

        device = _h7152()
        coordinator = MagicMock()
        coordinator.devices = {DEVICE_ID: device}
        coordinator.mqtt_client = None
        coordinator.get_state.return_value = None
        coordinator.is_bff_leak_sensor.return_value = False
        coordinator.leak_sensors = {}

        entry = MagicMock()
        entry.runtime_data = coordinator
        entry.entry_id = "test_entry"

        added: list = []
        await async_setup_entry(MagicMock(), entry, added.extend)

        temp_entities = [e for e in added if isinstance(e, GoveeTemperatureSensor)]
        assert len(temp_entities) == 1
