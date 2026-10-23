"""Credential redaction tests."""

from argparse import Namespace

from core.wandb_utils import redact_secrets


def test_redact_secrets_masks_credentials():
    args = Namespace(
        wandb_api_key='local-secret',
        token='abc',
        password='pw',
        normal_value=3,
        empty_token='',
    )
    safe = redact_secrets(args)
    assert safe['wandb_api_key'] == '***'
    assert safe['token'] == '***'
    assert safe['password'] == '***'
    assert safe['normal_value'] == 3
    assert safe['empty_token'] == ''
