from ..lib_omost.greedy_encode import greedy_partition, CLIPTokens


def test_greedy_partition():
    """Test with Omost repo example."""
    items = [
        CLIPTokens(clip_l_tokens=[i] * length)
        for i, length in enumerate([25, 35, 5, 60, 15, 25])
    ]
    bags = greedy_partition(items, max_sum=70)
    assert bags == [
        [items[0], items[1], items[2]],
        [items[3]],
        [items[4], items[5]],
    ]
