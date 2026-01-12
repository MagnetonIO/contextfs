"""ContextFS End-to-End Encryption Module.

Provides client-side encryption for memory content before syncing to the cloud.
The server never sees plaintext data.
"""

from contextfs.encryption.client_crypto import (
    ClientCrypto,
    decrypt_content,
    derive_encryption_key,
    derive_encryption_key_base64,
    encrypt_content,
)

__all__ = [
    "ClientCrypto",
    "decrypt_content",
    "derive_encryption_key",
    "derive_encryption_key_base64",
    "encrypt_content",
]
