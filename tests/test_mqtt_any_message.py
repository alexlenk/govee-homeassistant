"""Regression tests for GoveeAwsIotClient's on_any_message hook.

on_raw_message (added for the H7152 humidity investigation) still missed one
shape: a "msg"-wrapped payload lacking a top-level "device"+"state" pair
(e.g. a command-accepted ack, or a read-response using a different envelope)
is silently dropped by _handle_message's own unwrap-or-return logic before
on_raw_message's call site is ever reached. on_any_message fires before any
parsing/filtering at all, so nothing is invisible regardless of shape.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.govee.api.auth import GoveeIotCredentials
from custom_components.govee.api.mqtt import GoveeAwsIotClient


def _make_client(on_any_message=None, on_state_update=None) -> GoveeAwsIotClient:
    creds = GoveeIotCredentials(
        token="t",
        refresh_token="r",
        account_topic="GA/account",
        iot_cert="cert",
        iot_key="key",
        iot_ca=None,
        client_id="cid",
        endpoint="endpoint",
    )
    return GoveeAwsIotClient(
        creds,
        on_state_update=on_state_update or MagicMock(),
        on_any_message=on_any_message,
    )


def _msg(payload: bytes) -> MagicMock:
    m = MagicMock()
    m.payload = payload
    m.topic = "GA/account"
    return m


class TestOnAnyMessage:
    @pytest.mark.asyncio
    async def test_fires_for_ordinary_status_push(self):
        cb = MagicMock()
        client = _make_client(on_any_message=cb)
        payload = b'{"device":"AA:BB","state":{"onOff":1,"brightness":100}}'

        await client._handle_message(_msg(payload))

        cb.assert_called_once()
        topic, payload_str = cb.call_args.args
        assert topic == "GA/account"
        assert payload_str == payload.decode()

    @pytest.mark.asyncio
    async def test_fires_for_msg_wrapped_ack_that_on_raw_message_never_sees(self):
        """The exact shape observed live: a "msg"-wrapped ptReal ack with no
        top-level "device"/"state" pair — _handle_message's unwrap logic
        drops this before on_raw_message's call site is reached at all.
        """
        raw_cb = MagicMock()
        any_cb = MagicMock()
        client = _make_client(on_any_message=any_cb)
        client._on_raw_message = raw_cb
        payload = (
            b'{"msg": {"cmd": "ptReal", "data": {"command": [], "device": "AA:BB", '
            b'"sku": "H7152"}, "cmdVersion": 0, "transaction": "v_123", "type": 1}}'
        )

        await client._handle_message(_msg(payload))

        # on_raw_message never fires — the message is dropped before reaching it.
        raw_cb.assert_not_called()
        # on_any_message fires anyway, with the untouched payload.
        any_cb.assert_called_once()
        topic, payload_str = any_cb.call_args.args
        assert topic == "GA/account"
        assert payload_str == payload.decode()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_break_message_handling(self):
        cb = MagicMock(side_effect=RuntimeError("boom"))
        state_cb = MagicMock()
        client = _make_client(on_any_message=cb, on_state_update=state_cb)
        payload = b'{"device":"AA:BB","state":{"onOff":1}}'

        await client._handle_message(_msg(payload))

        cb.assert_called_once()
        state_cb.assert_called_once()

    @pytest.mark.asyncio
    async def test_none_callback_is_a_noop(self):
        client = _make_client(on_any_message=None)
        payload = b'{"device":"AA:BB","state":{"onOff":1}}'

        await client._handle_message(_msg(payload))  # must not raise
