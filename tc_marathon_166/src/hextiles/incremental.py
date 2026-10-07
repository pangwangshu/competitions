from hextiles.geometry import DC, DR, S, in_grid, partner_at, rotation_distance


class PathRecord:
    __slots__ = ("partner", "length", "bonus_count", "matched", "tiles")

    def __init__(self, partner, length, bonus_count, matched, tiles):
        self.partner = partner
        self.length = length
        self.bonus_count = bonus_count
        self.matched = matched
        self.tiles = tiles


class FlipResult:
    __slots__ = (
        "flips",
        "evaluation",
        "new_records",
        "matched_paths",
        "total_path_score",
        "move_count",
    )

    def __init__(self, flips, evaluation, new_records, matched_paths, total_path_score, move_count):
        self.flips = flips
        self.evaluation = evaluation
        self.new_records = new_records
        self.matched_paths = matched_paths
        self.total_path_score = total_path_score
        self.move_count = move_count


def trace_path(state, exits, exit_ids, start_exit):
    """Trace one path forward from start_exit through the current grid.

    Returns (end_exit, length, bonus_count, tiles_visited). end_exit is
    None if the trace revisits an already-visited (row, column, entry_edge)
    without reaching a border exit -- provably unreachable for a
    border-anchored trace (every tile has degree 2 at the interior-edge
    level, every border exit degree 1, so a walk starting at a border exit
    is always a simple path to another border exit) but kept as a
    defensive fallback rather than assuming it away.
    """
    row, column, entry_edge = exits[start_exit]
    path_length = 0
    bonus_count = 0
    visited_bonus_tiles = set()
    visited_entries = set()
    tiles_visited = set()
    while True:
        entry = (row, column, entry_edge)
        if entry in visited_entries:
            return None, path_length, bonus_count, frozenset(tiles_visited)
        visited_entries.add(entry)
        tiles_visited.add((row, column))
        path_length += 1
        if (row, column) in state.bonus and (row, column) not in visited_bonus_tiles:
            visited_bonus_tiles.add((row, column))
            bonus_count += 1

        exit_edge = partner_at(entry_edge, state.grid[row][column])
        next_row = row + DR[exit_edge]
        next_column = column + DC[exit_edge]
        if not in_grid(next_row, next_column, state.n, state.w):
            end_exit = exit_ids[(row, column, exit_edge)]
            return end_exit, path_length, bonus_count, frozenset(tiles_visited)

        row = next_row
        column = next_column
        entry_edge = (exit_edge + 3) % S


class IncrementalEvaluator:
    """Maintains the same (score, matched_paths, total_path_score, -move_count)
    tuple as solver.evaluate_grid, updated incrementally as tiles flip
    instead of by a full O(W^2) re-trace.

    Correctness invariant: flipping a set of tiles can only change the
    outcome of a path-trace for exits whose CURRENT trace visits one of
    those tiles (every other tile's hop behavior is unchanged, so any
    trace whose prefix never reaches a flipped tile is identical before
    and after). `tile_owners` stores BOTH endpoints of every live path
    against every tile it touches, so the set of possibly-affected exits
    for a flip is exactly the union of `tile_owners` over the flipped
    tiles -- no exit outside that set can change.
    """

    def __init__(self, state, exits, exit_ids, initial_grid, target_matches):
        self.state = state
        self.exits = exits
        self.exit_ids = exit_ids
        self.initial_grid = initial_grid
        self.target_matches = target_matches
        self.path_of = {}
        self.tile_owners = {}
        self.matched_paths = 0
        self.total_path_score = 0
        self.move_count = 0
        self._full_rebuild()

    def _register(self, exit_id, record):
        self.path_of[exit_id] = record
        for tile in record.tiles:
            self.tile_owners.setdefault(tile, set()).add(exit_id)

    def _full_rebuild(self):
        self.path_of = {}
        self.tile_owners = {}
        self.matched_paths = 0
        self.total_path_score = 0
        visited = set()
        for start_exit in range(len(self.exits)):
            if start_exit in visited:
                continue
            end_exit, length, bonus_count, tiles = trace_path(
                self.state, self.exits, self.exit_ids, start_exit
            )
            visited.add(start_exit)
            if end_exit is None:
                self._register(start_exit, PathRecord(None, length, bonus_count, False, tiles))
                continue
            visited.add(end_exit)
            matched = self.target_matches[start_exit] == end_exit
            self._register(start_exit, PathRecord(end_exit, length, bonus_count, matched, tiles))
            self._register(end_exit, PathRecord(start_exit, length, bonus_count, matched, tiles))
            if matched:
                self.matched_paths += 1
                self.total_path_score += length * (bonus_count + 1)

        self.move_count = sum(
            rotation_distance(self.initial_grid[row][column], self.state.grid[row][column])
            for row in range(self.state.w)
            for column in range(self.state.w)
            if self.initial_grid[row][column] >= 0
        )

    def evaluation(self):
        score = self.matched_paths * (self.total_path_score - self.move_count * self.state.m)
        return (max(score, 0), self.matched_paths, self.total_path_score, -self.move_count)

    def preview_flips(self, flips):
        """Compute the evaluation tuple that would result from applying
        `flips` (a {(row, column): new_orientation} dict), WITHOUT
        committing -- state.grid and the cached bookkeeping are left
        exactly as they were. Pass the returned FlipResult to
        commit_flips to apply it for real without redoing the retrace.
        """
        old_orientations = {position: self.state.grid[position[0]][position[1]] for position in flips}

        affected = set()
        for tile in flips:
            affected |= self.tile_owners.get(tile, set())

        matched_paths = self.matched_paths
        total_path_score = self.total_path_score
        seen_pairs = set()
        for exit_id in affected:
            record = self.path_of[exit_id]
            pair_key = frozenset((exit_id, record.partner)) if record.partner is not None else exit_id
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if record.matched:
                matched_paths -= 1
                total_path_score -= record.length * (record.bonus_count + 1)

        move_count = self.move_count
        for (row, column), new_orientation in flips.items():
            old_orientation = old_orientations[(row, column)]
            move_count += rotation_distance(self.initial_grid[row][column], new_orientation)
            move_count -= rotation_distance(self.initial_grid[row][column], old_orientation)
            self.state.grid[row][column] = new_orientation

        # Frontier = affected primaries plus their current partners. With
        # symmetric tile_owners this expansion is redundant in principle
        # (both endpoints already share the same touched-tile set, so both
        # land in `affected` directly) -- kept anyway as a cheap, defensive
        # second path to the same set.
        frontier = set(affected)
        for exit_id in affected:
            partner = self.path_of[exit_id].partner
            if partner is not None:
                frontier.add(partner)

        new_records = {}
        claimed = set()
        for exit_id in sorted(frontier):
            if exit_id in claimed:
                continue
            end_exit, length, bonus_count, tiles = trace_path(
                self.state, self.exits, self.exit_ids, exit_id
            )
            claimed.add(exit_id)
            if end_exit is None:
                new_records[exit_id] = PathRecord(None, length, bonus_count, False, tiles)
                continue
            matched = self.target_matches[exit_id] == end_exit
            new_records[exit_id] = PathRecord(end_exit, length, bonus_count, matched, tiles)
            if end_exit in frontier and end_exit not in claimed:
                new_records[end_exit] = PathRecord(exit_id, length, bonus_count, matched, tiles)
                claimed.add(end_exit)
            if matched:
                matched_paths += 1
                total_path_score += length * (bonus_count + 1)

        for (row, column), old_orientation in old_orientations.items():
            self.state.grid[row][column] = old_orientation

        score = max(matched_paths * (total_path_score - move_count * self.state.m), 0)
        evaluation = (score, matched_paths, total_path_score, -move_count)
        return FlipResult(dict(flips), evaluation, new_records, matched_paths, total_path_score, move_count)

    def commit_flips(self, result):
        """Permanently apply a FlipResult produced by preview_flips."""
        for (row, column), new_orientation in result.flips.items():
            self.state.grid[row][column] = new_orientation

        for exit_id, new_record in result.new_records.items():
            old_record = self.path_of.get(exit_id)
            if old_record is not None:
                for tile in old_record.tiles:
                    owners = self.tile_owners.get(tile)
                    if owners is not None:
                        owners.discard(exit_id)
                        if not owners:
                            del self.tile_owners[tile]
            self.path_of[exit_id] = new_record
            for tile in new_record.tiles:
                self.tile_owners.setdefault(tile, set()).add(exit_id)

        self.matched_paths = result.matched_paths
        self.total_path_score = result.total_path_score
        self.move_count = result.move_count

    def apply_flips(self, flips):
        """Preview + immediately commit. Returns the evaluation tuple."""
        result = self.preview_flips(flips)
        self.commit_flips(result)
        return result.evaluation

    def pair_tiles(self, first_exit, second_exit):
        """Tiles touched by either exit's CURRENT trace. If the two exits are
        currently each other's trace-partner, both records already reference
        the identical frozenset object, so this is just that one path's tile
        set; otherwise it's the union of two independent current traces.
        Always well-defined -- every exit has a PathRecord from __init__
        onward, regardless of match status.
        """
        return self.path_of[first_exit].tiles | self.path_of[second_exit].tiles
