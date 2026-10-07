S = 6

# direction order matching the tester: top-right, right, bottom-right,
# bottom-left, left, top-left
DR = [-1, 0, 1, 1, 0, -1]
DC = [1, 1, 0, -1, -1, 0]

# chords of a tile at rotation 0: edge pairs joined by a segment
CONNECTIONS = [(0, 5), (1, 3), (2, 4)]


def grid_width(n):
    return 2 * n - 1


def get_start(r, n, w):
    if r < n:
        return n - 1 - r
    return 0


def get_end(r, n, w):
    if r < n:
        return w - 1
    return w + n - r - 2


def in_grid(r, c, n, w):
    return 0 <= r < w and get_start(r, n, w) <= c <= get_end(r, n, w)


def tile_chord(rotation, path):
    a, b = CONNECTIONS[path]
    return (a + rotation) % S, (b + rotation) % S


def partner_at(entry_edge, orientation):
    for first_edge, second_edge in CONNECTIONS:
        first_edge = (first_edge + orientation) % S
        second_edge = (second_edge + orientation) % S
        if first_edge == entry_edge:
            return second_edge
        if second_edge == entry_edge:
            return first_edge
    raise ValueError("invalid tile edge")


def rotation_distance(current, target):
    clockwise = (target - current) % S
    return min(clockwise, S - clockwise)


def minimum_rotation(entry_edge, exit_edge, orientation):
    best = S + 1
    for target_orientation in range(S):
        if partner_at(entry_edge, target_orientation) == exit_edge:
            best = min(best, rotation_distance(orientation, target_orientation))
    return best
