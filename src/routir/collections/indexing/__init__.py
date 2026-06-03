"""Random-access readers and on-disk index machinery for collection views.

Today: byte-offset maps for plain JSONL (``offset_file.OffsetFile``) and
shard/offset-encoded IDs for MSMARCO v2.1 (``offset_file.MSMARCOSegOffset``).
Future (PR5b): tar-member offset indexes (``.taridx``).
"""
