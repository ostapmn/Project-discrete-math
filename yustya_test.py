"""Pages directions"""
def pages_directions(pages):
    """
    Converts inout dict into a list of list with a links directions.
    """
    pages_list = []
    for key, values in pages.items():
        if len(values) == 1:
            values = str(values).lstrip("['").rstrip("']")
            pages_list.append([key, values])
        elif len(values) > 1:
            for one_val in values:
                pages_list.append([key, one_val])
    pages_list = sorted(pages_list)
    return pages_list

print(pages_directions({'a': ['b', 'c'], 'b': ['c'], 'c': ['a', 'b']}))