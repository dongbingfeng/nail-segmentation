#!/usr/bin/env python3
"""Download SAM vit_l model checkpoint"""

import asyncio
import aiohttp
import os
from pathlib import Path


async def download_sam_model():
    models_dir = Path('./models')
    models_dir.mkdir(exist_ok=True)
    
    url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
    filename = 'sam_vit_l_0b3195.pth'
    filepath = models_dir / filename
    
    if filepath.exists():
        size_mb = filepath.stat().st_size // (1024 * 1024)
        print(f'Model {filename} already exists ({size_mb} MB)')
        return str(filepath)
    
    print(f'Downloading SAM vit_l model from {url}...')
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(filepath, 'wb') as f:
                        async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                mb_downloaded = downloaded // (1024*1024)
                                mb_total = total_size // (1024*1024)
                                print(f'Progress: {percent:.1f}% ({mb_downloaded}/{mb_total} MB)', end='\r')
                    
                    final_size = filepath.stat().st_size // (1024 * 1024)
                    print(f'\nSuccessfully downloaded {filename} ({final_size} MB)')
                    return str(filepath)
                else:
                    print(f'Failed to download: HTTP {response.status}')
                    return None
                    
    except Exception as e:
        print(f'Download failed: {e}')
        if filepath.exists():
            filepath.unlink()  # Remove partial download
        return None


if __name__ == "__main__":
    result = asyncio.run(download_sam_model())
    if result:
        print(f"SAM model ready at: {result}")
    else:
        print("Failed to download SAM model")
        exit(1) 