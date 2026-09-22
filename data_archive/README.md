# data_archive — compressed shards of the evaluation data

The two data directories needed for evaluation are large (hundreds of MB combined) and cannot be committed as-is because GitHub rejects files over 100 MB. This folder stores them as **compressed, split shards**, each smaller than 100 MB.

| Restored directory | Contents                                                     | Unpacked size | Consumed by |
| ------------------ | ------------------------------------------------------------ | ------------- | ----------- |
| `sandbox_cache/`   | Real tool-call caches (`*_cache.json`, ~470K entries, 10 tools) | ~880 MB       | Inference   |
| `poi2category/`    | POI metadata `all_poi_by_city.json` (392 cities)             | ~360 MB       | Evaluation  |

Shards total about 260 MB and unpack to about 1.2 GB. Please reserve enough disk space.

## Layout

```
data_archive/
├── sandbox_cache/
│   └── sandbox_cache.tar.gz.part.aa, .ab, ...
├── poi2category/
│   └── poi2category.tar.gz.part.aa, .ab, ...
├── manifest.sha256    # sha256 of the original files (checked after restore)
└── archive.sha256     # sha256 of each shard (checked before unpacking)
```
