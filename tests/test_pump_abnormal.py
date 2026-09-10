"""Pump-fault status for pump-model dehumidifiers (H7152 "Max") — issue #114 follow-up.

Reverse-engineered from a live clogged-drain capture on a real H7152: neither
the OpenAPI event-push channel nor the flat MQTT ``state`` keys carried any
trace of the fault (confirmed empty across two captures, two minutes apart,
while the app showed "Pump Abnormality"). The only signal is byte offset 11
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
