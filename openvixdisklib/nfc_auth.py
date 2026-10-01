# Copyright 2026 Cloudbase Solutions Srl
# All Rights Reserved.

"""VDDK-compatible vSphere NFC authentication.

VixDiskLib_ConnectEx / Open authenticate in two stages:

1. SOAP login to vCenter (or ESXi) and an internal NfcService call that
   returns a one-time vim.HostServiceTicket.
2. A TLS session to the ESXi authd daemon on TCP 902, completed with the
   ticket's sessionId and service name.

pyVim / pyVmomi are used for every public VIM operation (login, inventory,
HostServiceTicket). NfcService is not in the public WSDL, so it is registered
with pyVmomi's type system and invoked through the same SOAP stub.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import socket
import ssl
import time
import types
from dataclasses import dataclass

from pyVim.connect import Disconnect, SmartConnect
from pyVmomi import vim, vmodl
from pyVmomi.VmomiSupport import F_OPTIONAL, CreateManagedType, GetVmodlType

AUTHD_DEFAULT_PORT = 902
_SHA256_BANNER = "SHA256 supported"
_NFC_TYPES_REGISTERED = False
LOG = logging.getLogger(__name__)
_NFC_SERVICE_MOID_RE = re.compile(r"<nfcService[^>]*>([^<]+)</nfcService>")
_TASK_POLL_S = 0.5
_TASK_TIMEOUT_S = 300


def _ssl_client_context(verify: bool = True, legacy: bool = False) -> ssl.SSLContext:
    """Return a client TLS context built with public ``ssl`` APIs.

    Args:
        verify: When False, skip hostname checks and certificate
            validation.
        legacy: When True, allow the SHA-1 certificates ESXi 6.5 and
            6.7.0 authd present. OpenSSL 3's default security level
            rejects that handshake. ESXi 7 and 8 succeed with
            ``legacy=False``.
    """
    context = ssl.create_default_context()
    if not verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    if legacy:
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers("DEFAULT:@SECLEVEL=1")
    return context


def _register_nfc_types() -> None:
    """Register internal vim.NfcService methods on the pyVmomi type map."""
    global _NFC_TYPES_REGISTERED
    if _NFC_TYPES_REGISTERED:
        return
    with contextlib.suppress(Exception):
        GetVmodlType("vim.NfcService")
        _NFC_TYPES_REGISTERED = True
        return

    # Method: (vmodlName, wsdlName, version, params, result, privilege, faults)
    # Param:  (name, type, version, flags, privilege); flags 0 = required
    # Result: (flags, vmodlType, wsdlType)
    CreateManagedType(
        "vim.NfcService",  # vmodl name
        "NfcService",  # WSDL name
        "vmodl.ManagedObject",  # parent
        "vim.version.version1",  # version
        [],  # properties
        [  # methods
            (
                "getVmFiles",  # vmodl method
                "NfcGetVmFiles",  # WSDL method
                "vim.version.version1",
                (("vm", "vim.VirtualMachine", "vim.version.version1", 0, None),),
                (0, "vim.HostServiceTicket", "vim.HostServiceTicket"),  # result
                None,  # privilege
                None,  # faults
            ),
            (
                "randomAccessOpen",  # vmodl method
                "NfcRandomAccessOpenDisk",  # WSDL method
                "vim.version.version1",
                (
                    ("vm", "vim.VirtualMachine", "vim.version.version1", 0, None),
                    ("diskDeviceKey", "int", "vim.version.version1", 0, None),
                    (
                        "hostForAccess",
                        "vim.HostSystem",
                        "vim.version.version1",
                        F_OPTIONAL,  # flags
                        None,  # privilege
                    ),
                ),
                (0, "vim.HostServiceTicket", "vim.HostServiceTicket"),  # result
                None,  # privilege
                None,  # faults
            ),
            (
                "randomAccessOpenReadonly",  # vmodl method
                "NfcRandomAccessOpenReadonly",  # WSDL method
                "vim.version.version1",
                (
                    ("vm", "vim.VirtualMachine", "vim.version.version1", 0, None),
                    ("diskDeviceKey", "int", "vim.version.version1", 0, None),
                    (
                        "hostForAccess",
                        "vim.HostSystem",
                        "vim.version.version1",
                        F_OPTIONAL,  # flags
                        None,  # privilege
                    ),
                ),
                (0, "vim.HostServiceTicket", "vim.HostServiceTicket"),  # result
                None,  # privilege
                None,  # faults
            ),
            (
                "getServerNfcLibVersion",  # vmodl method
                "NfcGetServerNfcLibVersion",  # WSDL method
                "vim.version.version1",
                (("hostForAccess", "vim.HostSystem", "vim.version.version1", 0, None),),
                (0, "int", "int"),  # result
                None,  # privilege
                None,  # faults
            ),
        ],
    )
    _NFC_TYPES_REGISTERED = True


def _nfc_service_moid(si: vim.ServiceInstance) -> str:
    """Return the NfcService moref via the internal ``RetrieveInternalContent`` call.

    vCenter and a bare ESXi host disagree on this moref (``nfcService`` vs.
    ``ha-nfc-service``); VDDK resolves it dynamically instead of assuming
    vCenter's name, which is why OpenVixDiskLib must too. The response also
    carries ~20 other undocumented managed-object refs (agent manager, disk
    manager, and so on) that aren't worth registering with pyVmomi's type
    system just to read one field, so the call is issued as raw SOAP over
    the existing authenticated connection and only ``nfcService`` is pulled
    out of the XML.
    """
    stub = si._stub
    info = types.SimpleNamespace(
        wsdlName="RetrieveInternalContent", version=stub.version, params=()
    )
    request = stub.SerializeRequest(si, info, ())
    headers = {
        "Cookie": stub.cookie,
        "SOAPAction": stub.versionId,
        "Content-Type": "text/xml; charset=utf-8",
    }
    conn = stub.GetConnection()
    try:
        conn.request("POST", stub.path, request, headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")
    finally:
        stub.ReturnConnection(conn)
    match = _NFC_SERVICE_MOID_RE.search(body)
    if response.status != 200 or not match:
        raise RuntimeError(
            f"RetrieveInternalContent (status {response.status}) had no nfcService moref"
        )
    return match.group(1)


def nfc_service(si: vim.ServiceInstance) -> vim.NfcService:
    """Return the vCenter/ESXi NfcService managed object on ``si``'s SOAP stub.

    Args:
        si: An authenticated ServiceInstance from pyVim.connect.SmartConnect.
    """
    _register_nfc_types()
    nfc_cls = GetVmodlType("vim.NfcService")
    return nfc_cls(_nfc_service_moid(si), si._stub)


def connect_vim(
    host: str,
    username: str,
    password: str,
    port: int = 443,
    thumbprint: str | None = None,
    allow_untrusted: bool = False,
) -> vim.ServiceInstance:
    """Login to vCenter or ESXi using pyVim.connect.SmartConnect.

    Args:
        host: vCenter or ESXi hostname/IP.
        username: VIM user name.
        password: VIM password.
        port: HTTPS port, usually 443.
        thumbprint: Optional SHA-1 SSL thumbprint of the management endpoint.
            When set, the peer certificate is pinned to this digest and the
            system CA store is not used. pyVmomi's version-discovery GET
            does not pin, so a self-signed vCenter fails CA verification
            before SOAP login unless that handshake is skipped after the
            pin check.
        allow_untrusted: If True, skip certificate validation.
    """
    if thumbprint and not allow_untrusted:
        peer = get_ssl_cert_thumbprint(host, port)
        if _normalize_thumbprint(peer) != _normalize_thumbprint(thumbprint):
            raise ConnectionError(
                f"management SSL thumbprint mismatch: got {peer}, expected {thumbprint}"
            )
    skip_ca = allow_untrusted or bool(thumbprint)
    ssl_context = _ssl_client_context(verify=False) if skip_ca else None
    return SmartConnect(
        host=host,
        user=username,
        pwd=password,
        port=port,
        thumbprint=thumbprint,
        sslContext=ssl_context,
        disableSslCertValidation=skip_ca,
    )


def _virtual_disk_key(vm: vim.VirtualMachine, disk_path: str) -> int:
    """Return the VirtualDisk device key for ``disk_path``.

    ``disk_path`` may be the currently attached leaf or any parent in
    that disk's snapshot delta chain (``backing.parent``). After a
    snapshot, the VM's hardware points at the new leaf (for example
    ``…-000008.vmdk``) while VDDK Open still uses the snapshot file
    (``…-000007.vmdk``). Both share the same ``VirtualDisk.key``.
    """
    for device in vm.config.hardware.device:
        if not isinstance(device, vim.vm.device.VirtualDisk):
            continue
        backing = getattr(device, "backing", None)
        while backing is not None:
            if getattr(backing, "fileName", None) == disk_path:
                return device.key
            backing = getattr(backing, "parent", None)
    raise ValueError(f"VMDK path {disk_path!r} is not attached to {vm._moId}")


def get_nfc_ticket(
    si: vim.ServiceInstance,
    vm: vim.VirtualMachine,
    disk_device_key: int | None = None,
    host_for_access: vim.HostSystem | None = None,
    read_only: bool = True,
    disk_path: str | None = None,
) -> vim.HostServiceTicket:
    """Return a one-time NFC HostServiceTicket for ``vm``.

    Matches VDDK: ``NfcGetVmFiles`` when only the VM is known (read-only),
    ``NfcRandomAccessOpenReadonly`` / ``NfcRandomAccessOpenDisk`` when a
    virtual disk device key (or datastore path) is supplied.

    Args:
        si: Authenticated ServiceInstance.
        vm: Target virtual machine.
        disk_device_key: Optional VirtualDisk.device key (for example 2000).
        host_for_access: Host that should serve NFC; defaults to the VM's host.
        read_only: When False, request a writable ticket (needs a disk).
        disk_path: Datastore path used to resolve ``disk_device_key``.
    """
    nfc = nfc_service(si)
    if read_only and disk_device_key is None and disk_path is None:
        return nfc.GetVmFiles(vm)
    if disk_device_key is None:
        if disk_path is None:
            raise ValueError("writable NFC tickets need disk_path or disk_device_key")
        disk_device_key = _virtual_disk_key(vm, disk_path)
    if host_for_access is None:
        host_for_access = vm.runtime.host
    if read_only:
        return nfc.RandomAccessOpenReadonly(vm, disk_device_key, host_for_access)
    return nfc.RandomAccessOpen(vm, disk_device_key, host_for_access)


def server_nfc_lib_version(
    si: vim.ServiceInstance, host: vim.HostSystem | None
) -> int | None:
    """Return ``NfcGetServerNfcLibVersion`` for ``host``.

    ESXi 8 answers 11. ESXi 7.0 answers 7. ESXi 6.7 answers 2.
    ESXi 6.5 answers 0. ESXi 6.0 does not implement the method;
    that is reported as 0 so the ESXi 8 ``PlainText`` handshake is
    skipped. Values below 11 skip that handshake, which ESXi 6.7
    and 6.0 close with ``SESSION_COMPLETE``. ESXi 6.0 then uses
    synchronous fssrvr I/O because its session-params reply does
    not advertise a version message.
    ``None`` means the query failed for another reason; callers keep
    the ESXi 8 NFC handshake in that case.

    Args:
        si: Authenticated ServiceInstance.
        host: Host that serves NFC, usually ``vm.runtime.host``.
    """
    if host is None:
        return None
    try:
        return int(nfc_service(si).GetServerNfcLibVersion(host))
    except vmodl.fault.InvalidRequest as exc:
        message = getattr(exc, "msg", None) or str(exc)
        if "NfcGetServerNfcLibVersion" in message:
            LOG.info(
                "Host has no NfcGetServerNfcLibVersion; "
                "using the pre-6.5 NFC handshake"
            )
            return 0
        LOG.warning("Could not read the server NFC library version: %s", exc)
        return None
    except Exception as exc:
        LOG.warning("Could not read the server NFC library version: %s", exc)
        return None


def _format_thumbprint(digest: bytes) -> str:
    return ":".join(f"{byte:02X}" for byte in digest)


def _sha1_thumbprint(der_cert: bytes) -> str:
    return _format_thumbprint(hashlib.sha1(der_cert).digest())


def _normalize_thumbprint(thumbprint: str) -> str:
    return thumbprint.replace(":", "").replace(" ", "").upper()


def get_ssl_cert_thumbprint(
    host: str,
    port: int = 443,
    digest_algorithm: str = "sha1",
    ssl_context: ssl.SSLContext | None = None,
    timeout: float = 30.0,
) -> str:
    """Return the TLS certificate thumbprint of ``host``:``port``.

    Reads the peer certificate in DER form and hashes it with ``hashlib``.
    The result is colon-separated uppercase hex (for example
    ``A5:AF:7D:…``), matching VDDK / pyVmomi SHA-1 thumbprints.

    Args:
        host: Hostname or IP of the TLS server.
        port: TLS port, usually 443.
        digest_algorithm: Hash name accepted by ``hashlib.new``. Default
            ``sha1`` is the format VDDK and pyVmomi expect.
        ssl_context: Optional SSL context. When omitted, a default client
            context is used with hostname checks and certificate
            validation disabled so a self-signed management certificate
            can still be read.
        timeout: Connect timeout in seconds.
    """
    if ssl_context is None:
        ssl_context = _ssl_client_context(verify=False)
    with (
        socket.create_connection((host, port), timeout=timeout) as sock,
        ssl_context.wrap_socket(sock, server_hostname=host) as ssock,
    ):
        cert = ssock.getpeercert(binary_form=True)
    if not cert:
        raise ConnectionError(f"no peer certificate from {host}:{port}")
    return _format_thumbprint(hashlib.new(digest_algorithm, cert).digest())


def _readline(sock: socket.socket) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("authd connection closed")
        buf += chunk
        if len(buf) > 4096:
            raise ConnectionError("oversized authd response")
    return buf.decode("ascii", "replace").rstrip("\r\n")


def _expect_code(line: str, code: str, what: str) -> str:
    if not line.startswith(code):
        raise ConnectionError(f"authd {what} failed: {line}")
    return line[len(code) :].lstrip()


def _connect_authd_tls(
    host: str, port: int, timeout: float
) -> tuple[ssl.SSLSocket, str]:
    """Connect to authd and finish the first TLS handshake.

    The default client context is tried first. ESXi 6.5 answers that
    ClientHello with a handshake failure, and the same TCP sequence is
    repeated with a legacy context. A failed handshake happens before
    ``SESSION``, so the NFC ticket is still unused.

    Returns:
        The TLS socket and the plaintext 220 banner. The socket has
        ``legacy_tls`` set when the legacy context was required.
    """
    last_error: ssl.SSLError | None = None
    for legacy in (False, True):
        raw = socket.create_connection((host, port), timeout=timeout)
        try:
            banner = _readline(raw)
            if not banner.startswith("220"):
                raise ConnectionError(f"unexpected authd banner: {banner}")
            ssock = _ssl_client_context(verify=False, legacy=legacy).wrap_socket(
                raw, server_hostname=host
            )
        except ssl.SSLError as exc:
            raw.close()
            last_error = exc
            if legacy:
                raise
            LOG.info(
                "authd TLS handshake failed (%s); retrying with legacy ciphers",
                exc,
            )
            continue
        except Exception:
            raw.close()
            raise
        ssock.legacy_tls = legacy  # type: ignore[attr-defined]
        return ssock, banner
    raise ConnectionError(f"authd TLS handshake failed: {last_error}")


def nfcssl_service_name(service: str) -> str:
    """Return the NFCSSL authd PROXY service for an NFC service name.

    VCenter tickets still report ``vpxa-nfc``. NBDSSL uses
    ``PROXY vpxa-nfcssl`` (or ``ha-nfcssl`` on a direct ESXi ticket).

    Args:
        service: Ticket ``service`` field, for example ``vpxa-nfc``.
    """
    if service.endswith("ssl"):
        return service
    return f"{service}ssl"


def connect_authd(
    ticket: vim.HostServiceTicket,
    allow_untrusted: bool = False,
    timeout: float = 30.0,
    nfc_ssl: bool = True,
    fallback_host: str | None = None,
) -> ssl.SSLSocket:
    """Complete the ESXi authd handshake using an NFC HostServiceTicket.

    Wire sequence captured from VDDK against authd on TCP 902:

    1. Read the plaintext 220 banner, then wrap the socket with TLS.
    2. SESSION <sessionId>
    3. BANNER
    4. THUMBPRINT_SHA2 PlainText, only when the banner advertises
       ``SHA256 supported`` (ESXi 8). ESXi 6.5, 6.7, and 7 omit that
       token and answer the command with ``530 Please login with USER
       and PASS``.
    5. PROXY <ticket.service>     (vpxa-nfc / nbd) or vpxa-nfcssl (nbdssl)

    ``THUMBPRINT_SHA2 PlainText`` is used for both transports when the
    banner allows it. NBDSSL is selected by the PROXY service name;
    after ``200 Connect ha-nfcssl`` a second TLS handshake is started
    in ``nfc_open``.

    TLS uses the default client context first, which ESXi 7 and 8
    accept. ESXi 6.5 and 6.7.0 authd reject that handshake, and the
    connect is retried with ``@SECLEVEL=1`` so their SHA-1
    certificates are allowed.
    The successful socket is marked ``legacy_tls`` for the NBDSSL
    wrap.

    Args:
        ticket: One-time ticket from get_nfc_ticket().
        allow_untrusted: If False, require the peer SHA-1 thumbprint to match
            ticket.sslThumbprint.
        timeout: Socket timeout in seconds.
        nfc_ssl: When True (the default), PROXY to the NFCSSL service
            used by nbdssl. Pass False for plaintext NFC (nbd).
        fallback_host: Host to dial when ``ticket.host`` is unset. A ticket
            issued directly by a bare ESXi host (no vCenter) omits ``host``
            entirely, since the authd endpoint is that same host; pass the
            VIM connection's host in that case.
    """
    host = ticket.host or fallback_host
    port = ticket.port or AUTHD_DEFAULT_PORT
    ssock, banner = _connect_authd_tls(host, port, timeout)
    sha256_supported = _SHA256_BANNER in banner

    try:
        if not allow_untrusted and ticket.sslThumbprint:
            der_cert = ssock.getpeercert(True)
            if not der_cert:
                raise ConnectionError(f"no peer certificate from {host}:{port}")
            peer = _sha1_thumbprint(der_cert)
            if _normalize_thumbprint(peer) != _normalize_thumbprint(
                ticket.sslThumbprint
            ):
                raise ConnectionError(
                    f"ESXi SSL thumbprint mismatch: got {peer}, "
                    f"expected {ticket.sslThumbprint}"
                )

        ssock.sendall(f"SESSION {ticket.sessionId}\r\n".encode("ascii"))
        # Trailing space is part of the BANNER command token used by authd.
        ssock.sendall(b"BANNER \r\n")
        _expect_code(_readline(ssock), "220", "BANNER")

        if sha256_supported:
            ssock.sendall(b"THUMBPRINT_SHA2 PlainText\r\n")
            _expect_code(_readline(ssock), "200", "THUMBPRINT_SHA2")
        else:
            LOG.info(
                "authd banner does not advertise SHA256; skipping THUMBPRINT_SHA2"
            )

        service = ticket.service or "vpxa-nfc"
        if nfc_ssl:
            service = nfcssl_service_name(service)
        ssock.sendall(f"PROXY {service}\r\n".encode("ascii"))
        _expect_code(_readline(ssock), "200", "PROXY")
        return ssock
    except Exception:
        ssock.close()
        raise


class NfcAuthSession:
    """Authenticated VIM session plus an authd/NFC TLS socket."""

    def __init__(
        self,
        si: vim.ServiceInstance,
        ticket: vim.HostServiceTicket,
        authd_sock: ssl.SSLSocket,
        nfc_ssl: bool = True,
    ) -> None:
        self.si = si
        self.ticket = ticket
        self.authd_sock = authd_sock
        self.nfc_ssl = nfc_ssl

    def close(self) -> None:
        """Close the authd socket and logout of the VIM session."""
        try:
            self.authd_sock.close()
        finally:
            Disconnect(self.si)

    def __enter__(self) -> NfcAuthSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def authenticate(
    host: str,
    username: str,
    password: str,
    vm_moref: str,
    port: int = 443,
    thumbprint: str | None = None,
    allow_untrusted: bool = False,
    disk_device_key: int | None = None,
    disk_path: str | None = None,
    read_only: bool = True,
    nfc_ssl: bool = True,
) -> NfcAuthSession:
    """Login to vSphere and complete NFC authd authentication for a VM.

    Args:
        host: vCenter or ESXi hostname/IP.
        username: VIM user name.
        password: VIM password.
        vm_moref: Virtual machine managed object id (for example ``vm-13098``).
        port: HTTPS port for VIM, usually 443.
        thumbprint: Optional SHA-1 thumbprint of the management endpoint.
        allow_untrusted: Skip TLS certificate checks when True.
        disk_device_key: Optional VirtualDisk device key; when omitted with
            ``read_only``, the VDDK ``NfcGetVmFiles`` ticket is used.
        disk_path: Datastore path used to resolve ``disk_device_key``.
        read_only: When False, request a writable ``NfcRandomAccessOpenDisk``
            ticket.
        nfc_ssl: When True (the default), complete authd with the NFCSSL
            PROXY service used by nbdssl. Pass False for nbd.
    """
    si = connect_vim(
        host,
        username,
        password,
        port=port,
        thumbprint=thumbprint,
        allow_untrusted=allow_untrusted,
    )
    try:
        vm = vim.VirtualMachine(vm_moref, si._stub)
        ticket = get_nfc_ticket(
            si,
            vm,
            disk_device_key=disk_device_key,
            disk_path=disk_path,
            read_only=read_only,
        )
        authd_sock = connect_authd(
            ticket, allow_untrusted=allow_untrusted, nfc_ssl=nfc_ssl, fallback_host=host
        )
    except Exception:
        Disconnect(si)
        raise
    return NfcAuthSession(si, ticket, authd_sock, nfc_ssl=nfc_ssl)


# --- Changed Block Tracking (CBT) ---
#
# VixDiskLib does not expose CBT itself: ``VixDiskLib_QueryAllocatedBlocks``
# reports which blocks are allocated (non-sparse) within a single
# NFC-opened disk, not which byte ranges changed between two points in
# time. Real backup tools get that from vSphere's public
# ``VirtualMachine.QueryChangedDiskAreas`` VIM call instead, used
# alongside VDDK/NFC reads. The helpers below are thin wrappers around
# that public pyVmomi call — no NFC reverse engineering was needed for
# them. See ``docs/cbt.md``.


@dataclass(frozen=True, slots=True)
class ChangedExtent:
    """One changed byte range, as returned by ``QueryChangedDiskAreas``."""

    start: int
    length: int


@dataclass(frozen=True, slots=True)
class ChangedDiskAreas:
    """Result of ``QueryChangedDiskAreas``, converted to plain dataclasses."""

    start_offset: int
    length: int
    changed_areas: tuple[ChangedExtent, ...]


def _wait_for_cbt_task(task: vim.Task):
    deadline = time.monotonic() + _TASK_TIMEOUT_S
    while task.info.state in (vim.TaskInfo.State.running, vim.TaskInfo.State.queued):
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for vSphere task {task}")
        time.sleep(_TASK_POLL_S)
    if task.info.state != vim.TaskInfo.State.success:
        raise RuntimeError(f"vSphere task failed: {task.info.error}")
    return task.info.result


def enable_change_tracking(vm: vim.VirtualMachine) -> None:
    """Enable CBT on ``vm``.

    Takes effect for writes from this point forward; it does not
    retroactively track earlier changes. A ``changeId`` for a disk only
    becomes available after the next snapshot or power cycle once this
    is set.
    """
    spec = vim.vm.ConfigSpec(changeTrackingEnabled=True)
    _wait_for_cbt_task(vm.ReconfigVM_Task(spec=spec))


def disk_change_id(vm: vim.VirtualMachine, device_key: int) -> str:
    """Return the current ``changeId`` for the disk with ``device_key``.

    Requires CBT to be enabled and at least one snapshot (or power
    cycle) to have happened since. Raises ``ValueError`` if the disk
    isn't found or has no ``changeId`` yet (CBT not active for it).
    """
    for device in vm.config.hardware.device:
        if isinstance(device, vim.vm.device.VirtualDisk) and device.key == device_key:
            change_id = getattr(device.backing, "changeId", None)
            if not change_id:
                raise ValueError(
                    f"disk {device_key} on {vm._moId} has no changeId yet "
                    "(enable CBT and take a snapshot first)"
                )
            return change_id
    raise ValueError(f"no VirtualDisk with device key {device_key} on {vm._moId}")


def query_changed_disk_areas(
    vm: vim.VirtualMachine,
    snapshot: vim.vm.Snapshot,
    device_key: int,
    change_id: str,
    start_offset: int = 0,
) -> ChangedDiskAreas:
    """Return byte ranges changed since ``change_id``, up to ``snapshot``.

    Thin wrapper around the public
    ``VirtualMachine.QueryChangedDiskAreas`` VIM call. ``change_id`` is
    the value from an earlier ``disk_change_id()`` call (or ``"*"`` for
    the entire disk, e.g. for an initial full backup). Extents are
    64 KiB-aligned in this lab's observations, but that granularity is
    server-defined and not part of this function's contract.
    """
    result = vm.QueryChangedDiskAreas(
        snapshot=snapshot,
        deviceKey=device_key,
        startOffset=start_offset,
        changeId=change_id,
    )
    return ChangedDiskAreas(
        start_offset=result.startOffset,
        length=result.length,
        changed_areas=tuple(
            ChangedExtent(start=extent.start, length=extent.length)
            for extent in result.changedArea
        ),
    )
