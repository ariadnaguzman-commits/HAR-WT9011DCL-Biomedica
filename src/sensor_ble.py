import asyncio
import struct
import time
from bleak import BleakClient, BleakScanner
from PyQt6.QtCore import QThread, pyqtSignal

try:
    from src.config import DEVICE_NAME_PREFIX, FS, PACKET_HEADER, PACKET_LEN, UUID_DATOS
except ImportError:
    from config import DEVICE_NAME_PREFIX, FS, PACKET_HEADER, PACKET_LEN, UUID_DATOS


def decodificar_paquete_wt9011(paquete: bytes) -> tuple[float, float, float, float, float, float] | None:
    """
    Decodifica un paquete de 20 bytes del sensor WitMotion WT9011DCL (encabezado 0x55 0x61).
    Devuelve (ax, ay, az, gx, gy, gz) en [g] y [deg/s].
    """
    if len(paquete) != PACKET_LEN or paquete[0:2] != PACKET_HEADER:
        return None

    ax_raw, ay_raw, az_raw, gx_raw, gy_raw, gz_raw = struct.unpack("<6h", paquete[2:14])

    ax = (ax_raw / 32768.0) * 16.0
    ay = (ay_raw / 32768.0) * 16.0
    az = (az_raw / 32768.0) * 16.0

    gx = (gx_raw / 32768.0) * 2000.0
    gy = (gy_raw / 32768.0) * 2000.0
    gz = (gz_raw / 32768.0) * 2000.0

    return (ax, ay, az, gx, gy, gz)


class BLEScannerWorker(QThread):
    """Hilo secundario para escanear dispositivos BLE cercanos sin bloquear la GUI."""

    devices_found = pyqtSignal(list)  # Lista de tuplas (nombre, direccion_mac)
    scan_error = pyqtSignal(str)

    def __init__(self, timeout: float = 4.0, parent=None):
        super().__init__(parent)
        self.timeout = timeout

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            encontrados = loop.run_until_complete(self._escanear())
            self.devices_found.emit(encontrados)
        except Exception as exc:
            self.scan_error.emit(f"Error al escanear Bluetooth: {exc}")
        finally:
            loop.close()

    async def _escanear(self) -> list[tuple[str, str]]:
        devices = await BleakScanner.discover(timeout=self.timeout)
        prioritarios = []
        otros = []

        for d in devices:
            nombre = d.name or "Dispositivo sin nombre"
            item = (nombre, d.address)
            if DEVICE_NAME_PREFIX.lower() in nombre.lower() or "wit" in nombre.lower():
                prioritarios.append(item)
            elif d.name:
                otros.append(item)

        return prioritarios + otros


class BLEStreamWorker(QThread):
    """Hilo de transmisión BLE en tiempo real desde el sensor WT9011DCL."""

    # Emite (tiempo_relativo_s, ax, ay, az, gx, gy, gz)
    sample_received = pyqtSignal(tuple)
    status_changed = pyqtSignal(str, bool)  # (mensaje, esta_conectado)

    def __init__(self, address: str, parent=None):
        super().__init__(parent)
        self.address = address
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._stream())
        except Exception as exc:
            self.status_changed.emit(f"Error BLE: {exc}", False)
        finally:
            loop.close()

    async def _stream(self) -> None:
        self.status_changed.emit(f"Conectando a {self.address}...", False)
        buffer_bytes = bytearray()
        t_inicio = None

        def notification_handler(_sender, data: bytearray) -> None:
            nonlocal t_inicio
            buffer_bytes.extend(data)

            while len(buffer_bytes) >= PACKET_LEN:
                idx = buffer_bytes.find(PACKET_HEADER)
                if idx == -1:
                    buffer_bytes.clear()
                    break
                if idx > 0:
                    del buffer_bytes[:idx]
                if len(buffer_bytes) < PACKET_LEN:
                    break

                paquete = bytes(buffer_bytes[:PACKET_LEN])
                del buffer_bytes[:PACKET_LEN]

                muestra = decodificar_paquete_wt9011(paquete)
                if muestra is not None:
                    ahora = time.perf_counter()
                    if t_inicio is None:
                        t_inicio = ahora
                    t_rel = round(ahora - t_inicio, 4)
                    self.sample_received.emit((t_rel, *muestra))

        async with BleakClient(self.address, timeout=10.0) as client:
            if not client.is_connected:
                self.status_changed.emit("No se pudo establecer conexión BLE.", False)
                return

            await client.start_notify(UUID_DATOS, notification_handler)
            self.status_changed.emit(f"Conectado a {self.address} ({FS} Hz)", True)

            while not self._stop_requested and client.is_connected:
                await asyncio.sleep(0.05)

            if client.is_connected:
                await client.stop_notify(UUID_DATOS)

        self.status_changed.emit("Sensor desconectado.", False)