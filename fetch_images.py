#!/usr/bin/env python3
import os
import argparse
import zarr
import asyncio
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import random
import time


async def fetch(i, url, session, failed,
                max_retries: int = 3,
                base_delay: float = 0.5,
                max_delay: float = 10.0):
    """
    Fetch URL with exponential backoff + full jitter.
    On ultimate failure, logs into `failed` and returns a blank image.
    """
    for attempt in range(1, max_retries + 1):
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    raise aiohttp.ClientResponseError(
                        status=404,
                        request_info=resp.request_info,
                        history=resp.history,
                        message="Not Found"
                    )
                resp.raise_for_status()
                data = await resp.read()
            img = Image.open(BytesIO(data)).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            return i, np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)

        except Exception as exc:
            # if last attempt, record failure
            if attempt == max_retries:
                failed.append((i, url, repr(exc)))
                return i, np.zeros((3, 256, 256), dtype=np.uint8)

            # otherwise back off with full jitter
            exp = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = random.uniform(0, exp)
            await asyncio.sleep(delay)

    # should never reach here
    failed.append((i, url, "unknown error"))
    return i, np.zeros((3, 256, 256), dtype=np.uint8)


async def download_all(df, images, batch_size, max_concurrency, failed):
    """
    Creates one aiohttp session and downloads in batches.
    Uses exponential backoff + jitter on each fetch.
    Logs simple progress when moving to each batch.
    """
    n = len(df)
    timeout   = ClientTimeout(total=10)
    connector = TCPConnector(limit=max_concurrency, force_close=True)

    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=connector) as session:

        num_batches = (n + batch_size - 1) // batch_size
        for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            idxs = list(range(start, end))
            urls = df['identifier'].iloc[start:end].tolist()

            print(f"[{time.strftime('%H:%M:%S')}] Starting batch {batch_idx}/{num_batches} "
                  f"(images {start}–{end-1})")

            tasks = [asyncio.create_task(fetch(i, url, session, failed))
                     for i, url in zip(idxs, urls)]

            batch_arr = np.zeros((end - start, 3, 256, 256), dtype=np.uint8)
            for coro in asyncio.as_completed(tasks):
                i, arr = await coro
                batch_arr[i - start] = arr

            images[start:end] = batch_arr

            print(f"[{time.strftime('%H:%M:%S')}] Finished batch {batch_idx}/{num_batches}")



def build_zarr_store(df: pd.DataFrame, zarr_path: str,
                     batch_size: int = 1000,
                     max_concurrency: int = 200):
    """
    Synchronous entrypoint: sets up Zarr, then runs asyncio for downloads.
    """
    os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
    n = len(df)
    root = zarr.open_group(zarr_path, mode='w')

    images = root.create_array(
        name='images',
        shape=(n, 3, 256, 256),
        chunks=(batch_size, 3, 256, 256),
        dtype='u1',
        overwrite=True,
    )
    gbif = root.create_array('gbifID', shape=(n,), dtype='i8', overwrite=True)
    gbif[:] = df['gbifID'].values.astype('i8')

    genus_str = df['genus'].fillna('').astype(str)
    maxlen = int(genus_str.str.len().max())
    genus = root.create_array('genus', shape=(n,), dtype=f'|S{maxlen}', overwrite=True)
    genus_bytes = (
        genus_str
        .str.encode('utf-8')
        .apply(lambda b: b.ljust(maxlen, b'\0')[:maxlen])
        .values
    )
    genus[:] = genus_bytes

    failed = []
    asyncio.run(download_all(df, images, batch_size, max_concurrency, failed))

    if failed:
        print(f"\nWarning: {len(failed)} images failed. Sample:")
        for i, url, err in failed[:5]:
            print(f"  [{i}] {url} → {err}")

    print(f"Zarr store built at {zarr_path}")


def main():
    parser = argparse.ArgumentParser(description="Download images to Zarr store.")
    parser.add_argument('csv_path',
                        help="CSV file with columns 'identifier','gbifID','genus'.")
    parser.add_argument('zarr_path',
                        help="Output Zarr directory path.")
    parser.add_argument('--sample-size', type=int, default=None,
                        help="Number of DataFrame rows to sample. Uses full DataFrame if None.")
    parser.add_argument('--batch-size', type=int, default=1000,
                        help="Number of images per download batch (and Zarr chunk).")
    parser.add_argument('--concurrency', type=int, default=200,
                        help="Max simultaneous HTTP connections.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.sample_size is not None:
        df = df.sample(args.sample_size, axis=0)

    build_zarr_store(df, args.zarr_path,
                     batch_size=args.batch_size,
                     max_concurrency=args.concurrency)


if __name__ == '__main__':
    main()