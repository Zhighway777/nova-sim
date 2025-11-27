def addr_to_llc_index(addr: int, cache_line_size: int, sets_per_slice: int, total_slices: int) -> int:
    """Simple helper that mimics the behaviour required by GCU Libra."""
    if total_slices <= 0:
        return 0
    line_addr = addr // cache_line_size
    if sets_per_slice <= 0:
        return line_addr % total_slices
    return (line_addr // sets_per_slice) % total_slices


__all__ = ["addr_to_llc_index"]
