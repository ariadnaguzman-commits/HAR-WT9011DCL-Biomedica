import asyncio
import math
import struct
import time

import pandas as pd
from bleak import BleakClient, BleakScanner

try:
    from src.config import (
        CLASES,
        DATA_RAW_DIR,
        DEFAULT_SENSOR_ADDRESS,
        DEFAULT_SENSOR_NAME,
        DISCARD_INITIAL_SECONDS,
        FS,
        PACKET_HEADER,
        PACKET_LEN,
        SENSOR_COLS,
        UUID_DATOS,
    )
except ImportError:
    from config import (
        CLASES,
        DATA_RAW_DIR,
        DEFAULT_SENSOR_ADDRESS,
        DEFAULT_SENSOR_NAME,
        DISCARD_INITIAL_SECONDS,
        FS,
        PACKET_HEADER,
        PACKET_LEN,
        SENSOR_COLS,
        UUID_DATOS,
    )

NOMBRE_SENSOR = DEFAULT_SENSOR_NAME
DIRECCION_SENSOR = DEFAULT_SENSOR_ADDRESS
FRECUENCIA_HZ = FS

MOVIMIENTOS = CLASES
COLUMNAS = SENSOR_COLS

SUJETO = "S07"
DURACION_S = {"caminar": 45, "brincar": 30, "subir_escaleras": 30}
DESCARTAR_INICIO_S = DISCARD_INITIAL_SECONDS

CARPETA_DATOS = DATA_RAW_DIR
CARPETA_DATOS.mkdir(parents=True, exist_ok=True)


# CONECTIVIDAD Y PRUEBA DE ADQUISICIÓN DE SEÑAL (BLE - WT9011DCL)

def decodificar(datos: bytearray) -> list[list[float]]:
    """Convierte los bytes enviados por el sensor en mediciones [ax, ay, az, gx, gy, gz]."""
    mediciones = []
    i = 0
    while i + PACKET_LEN <= len(datos):
        if datos[i : i + 2] == PACKET_HEADER:
            ax, ay, az, gx, gy, gz = struct.unpack("<6h", datos[i + 2 : i + 14])
            acc = [v / 32768 * 16 for v in (ax, ay, az)]      # en g
            giro = [v / 32768 * 2000 for v in (gx, gy, gz)]   # en °/s
            mediciones.append(acc + giro)
            i += PACKET_LEN
        else:
            i += 1
    return mediciones


async def escanear_sensores(timeout: int = 8) -> None:
    """Lista los sensores WitMotion cercanos con su intensidad de señal (RSSI)."""
    print(f"Escaneando dispositivos Bluetooth durante {timeout} s...")
    encontrados = await BleakScanner.discover(timeout=timeout, return_adv=True)
    for direccion, (d, anuncio) in encontrados.items():
        if d.name and d.name.startswith("WT"):
            print(f"{direccion}   {d.name}   señal: {anuncio.rssi} dBm")


async def buscar_sensor():
    """Busca el sensor BLE por dirección MAC o por nombre."""
    print("Buscando el sensor...")
    if DIRECCION_SENSOR:
        sensor = await BleakScanner.find_device_by_address(DIRECCION_SENSOR, timeout=30)
    else:
        sensor = await BleakScanner.find_device_by_name(NOMBRE_SENSOR, timeout=30)

    if sensor:
        print(f"Sensor encontrado: {sensor.name}  ({sensor.address})")
    else:
        raise RuntimeError(f"No encontré '{DIRECCION_SENSOR or NOMBRE_SENSOR}'")
    return sensor


async def leer_sensor(cliente: BleakClient, segundos: int, mostrar_cuenta: bool = False) -> pd.DataFrame:
    """Escucha notificaciones BLE hasta reunir `segundos * FRECUENCIA_HZ` muestras."""
    esperadas = segundos * FRECUENCIA_HZ
    limite_s = 2 * segundos + 5
    mediciones = []

    def al_recibir(_, datos):
        mediciones.extend(decodificar(datos))

    if not cliente.is_connected:
        await cliente.connect()

    await cliente.start_notify(UUID_DATOS, al_recibir)
    inicio = time.time()
    ultimo_mostrado = None
    try:
        while len(mediciones) < esperadas:
            if time.time() - inicio > limite_s:
                raise TimeoutError(
                    f"Solo llegaron {len(mediciones)} de {esperadas} muestras en {limite_s} s"
                )
            if mostrar_cuenta:
                restantes = math.ceil((esperadas - len(mediciones)) / FRECUENCIA_HZ)
                if restantes != ultimo_mostrado:
                    print(restantes, end=" ", flush=True)
                    ultimo_mostrado = restantes
            await asyncio.sleep(0.05)
    finally:
        if cliente.is_connected:
            await cliente.stop_notify(UUID_DATOS)

    return pd.DataFrame(mediciones[:esperadas], columns=COLUMNAS)


# PROTOCOLO DE VARIANTES Y GRABACIÓN DE TOMAS

VARIANTES = {
    "caminar": [
        ("normal", "Camina en línea recta a paso NORMAL"),
        ("rapido", "Camina en línea recta a paso RÁPIDO"),
        ("lento", "Camina en línea recta a paso LENTO"),
    ],
    "brincar": [
        ("bajo", "Brinca en el lugar con saltos BAJOS"),
        ("alto", "Brinca en el lugar con saltos ALTOS"),
        ("continuo", "Brinca en el lugar de forma CONTINUA, sin pausas"),
    ],
    "subir_escaleras": [
        ("normal", "Sube un tramo de escaleras a paso NORMAL"),
        ("rapido", "Sube un tramo de escaleras RÁPIDO"),
        ("lento", "Sube un tramo de escaleras LENTO"),
    ],
}

REPETICIONES = 3
filas = []
for repeticion in range(1, REPETICIONES + 1):
    for movimiento, variantes in VARIANTES.items():
        for variante, instruccion in variantes:
            filas.append((movimiento, variante, repeticion, instruccion, DURACION_S[movimiento]))

PROTOCOLO = pd.DataFrame(
    filas,
    columns=["movimiento", "variante", "repeticion", "instruccion", "duracion_s"],
)
PROTOCOLO.index += 1


def mostrar_protocolo() -> None:
    """Imprime el resumen y la tabla completa de las 27 tomas."""
    total_s = (PROTOCOLO["duracion_s"] + DESCARTAR_INICIO_S).sum()
    print(f"{len(PROTOCOLO)} tomas por sujeto, unos {total_s / 60:.0f} min de grabación (sin contar las pausas)")
    print(PROTOCOLO[["movimiento", "variante", "repeticion", "duracion_s"]])


async def grabar_toma(cliente: BleakClient, numero: int) -> None:
    """Graba la toma `numero` del PROTOCOLO y la guarda en `CARPETA_DATOS` (`data/raw/`)."""
    if numero not in PROTOCOLO.index:
        raise ValueError(f"Número de toma inválido: {numero}. Debe estar entre 1 y {len(PROTOCOLO)}.")

    toma = PROTOCOLO.loc[numero]
    segundos = int(toma.duracion_s) + DESCARTAR_INICIO_S

    print("=" * 70)
    print(f"  SUJETO {SUJETO}  -  TOMA {numero} de {len(PROTOCOLO)}  -  {toma.movimiento.upper()} ({toma.variante})")
    print(f"  {toma.instruccion}")
    print(f"  Duración: {toma.duracion_s} s")
    print("=" * 70)

    while True:
        input("Cuando estén listos (sensor ajustado en la cintura), presionen Enter...")

        if not cliente.is_connected:
            print("Reconectando...")
            await cliente.connect()

        for s in (3, 2, 1):
            print(f"{s}...", end=" ", flush=True)
            await asyncio.sleep(1)
        print("¡YA! Palmada y digan 'inicio'. Grabando:", end=" ", flush=True)

        try:
            df = await leer_sensor(cliente, segundos, mostrar_cuenta=True)
            break
        except TimeoutError as error:
            print(f"\n{error}. La toma no se guardó, repetir.\n")

    print("\n¡ALTO! Digan 'fin'.")

    # Descarta el primer segundo (palmada y arranque del movimiento)
    df = df.iloc[DESCARTAR_INICIO_S * FRECUENCIA_HZ :].reset_index(drop=True)

    esperadas = int(toma.duracion_s) * FRECUENCIA_HZ
    assert len(df) == esperadas, f"Se esperaban {esperadas} muestras y hay {len(df)}"

    df["movimiento"] = toma.movimiento

    archivo = CARPETA_DATOS / f"{SUJETO}_{numero:02d}_{toma.movimiento}_{toma.variante}_r{toma.repeticion}.csv"
    df.to_csv(archivo, index=False)
    print(f"Guardado: {archivo}  ({len(df)} muestras)\n")


def tomas_pendientes() -> list[int]:
    """Devuelve los números de toma (1..27) del sujeto actual que aún no tienen CSV en `CARPETA_DATOS`."""
    pendientes = []
    for numero, toma in PROTOCOLO.iterrows():
        archivo = CARPETA_DATOS / f"{SUJETO}_{numero:02d}_{toma.movimiento}_{toma.variante}_r{toma.repeticion}.csv"
        if not archivo.exists():
            pendientes.append(numero)
    return pendientes


async def main() -> None:
    mostrar_protocolo()

    sensor = await buscar_sensor()
    async with BleakClient(sensor) as cliente:
        print("Conectado:", cliente.is_connected)

        while True:
            pendientes = tomas_pendientes()
            print(f"\nSujeto: {SUJETO} | Tomas completadas: {len(PROTOCOLO) - len(pendientes)}/{len(PROTOCOLO)}")
            if pendientes:
                print(f"Siguiente toma recomendada: {pendientes[0]}")

            entrada = input(
                f"Ingresa el número de toma a grabar (1-{len(PROTOCOLO)}), "
                "'todas' para grabar las pendientes en orden, o '0' para salir: "
            ).strip().lower()

            if entrada == "0":
                break
            elif entrada == "todas":
                for num in pendientes:
                    await grabar_toma(cliente, num)
            elif entrada.isdigit() and 1 <= int(entrada) <= len(PROTOCOLO):
                await grabar_toma(cliente, int(entrada))
            else:
                print("Entrada no válida, intenta de nuevo.")


if __name__ == "__main__":
    asyncio.run(main())