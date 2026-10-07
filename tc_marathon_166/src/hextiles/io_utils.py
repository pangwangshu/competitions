def write_output(moves, stream):
    stream.write(f"{len(moves)}\n")
    for r, c, direction in moves:
        stream.write(f"{r} {c} {direction}\n")
    stream.flush()
