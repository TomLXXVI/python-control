import itertools
import sympy as sp
from .transfer_function import TransferFunction

TTransferFunction = sp.Function | sp.Integer | TransferFunction


class Node:

    def __init__(self, name: str) -> None:
        self.name = name
        self.entering_branches = []
        self.leaving_branches = []

    def __repr__(self):
        return self.name


class Branch:

    def __init__(
        self,
        name: str,
        tf: TTransferFunction | None,
        start_node_id: str,
        end_node_id: str,
        feedback: bool = False
    ) -> None:
        self.name = name
        self.tf = tf
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.feedback = feedback

    def __repr__(self):
        return self.name


class Path(list[Branch]):

    def __repr__(self):
        return ' ->- '.join(br.name for br in self)

    @property
    def gain(self) -> TTransferFunction:
        gain = self[0].tf
        for branch in self[1:]:
            gain *= branch.tf
        return gain

    def is_touching(self, other: 'Path') -> bool:
        """
        Checks if path `self` and path `other` have nodes in common.
        """
        nodes_self = []
        for branch in self:
            nodes_self.extend([branch.start_node_id, branch.end_node_id])
        nodes_self = set(nodes_self)

        nodes_other = []
        for branch in other:
            nodes_other.extend([branch.start_node_id, branch.end_node_id])
        nodes_other = set(nodes_other)

        if nodes_self.intersection(nodes_other):
            return True
        return False


class Loop(Path):

    def __eq__(self, other: 'Loop') -> bool:
        if len(self) != len(other):
            return False
        else:
            return all([
                True if branch in self else False
                for branch in other
            ])


class LoopList(list[Loop]):

    def __contains__(self, loop: Loop) -> bool:
        for loop_ in self:
            if loop_ == loop:
                return True
        return False


class SignalFlowGraph:

    class Branch(Branch):
        pass

    def __init__(self, start_node_id: str, end_node_id: str) -> None:
        """
        Creates a `SignalFlowGraph` object.

        Parameters
        ----------
        start_node_id:
            Unique name of the graph's start node.
        end_node_id:
            Unique name of the graph's end node.
        """
        self.branches: dict[str, Branch] = {}
        self.nodes = {
            start_node_id: Node(start_node_id),
            end_node_id: Node(end_node_id)
        }
        self.start_node = self.nodes[start_node_id]
        self.end_node = self.nodes[end_node_id]
        self._paths: list[Path] = []
        self._forward_paths: list[Path] | None = None
        self._loops: LoopList | None = None

    def add_branch(self, new_br: Branch) -> None:
        """
        Adds a new branch to the graph.
        """
        br = self.branches.setdefault(new_br.name, new_br)
        if br is not new_br:
            raise ValueError(f"Branch {new_br.name} was already added before.")
        else:
            start_node = self.nodes.setdefault(
                new_br.start_node_id,
                Node(new_br.start_node_id)
            )
            start_node.leaving_branches.append(new_br)
            end_node = self.nodes.setdefault(
                new_br.end_node_id,
                Node(new_br.end_node_id)
            )
            end_node.entering_branches.append(new_br)

    def _check_if_transfer_function(self) -> bool:
        branches = list(self.branches.values())
        if isinstance(branches[0], TransferFunction):
            return True
        return False

    def _search_paths(self) -> None:
        node = self.start_node
        path = Path()
        self._paths.append(path)
        self._recursive_path_search(node, path)

    def _recursive_path_search(self, node: Node, path: Path) -> None:
        while True:
            if len(node.leaving_branches) > 1:
                for branch in node.leaving_branches[1:]:
                    if branch not in path:
                        new_path = Path(path)
                        self._paths.append(new_path)
                        new_path.append(branch)
                        next_node = self.nodes[branch.end_node_id]
                        self._recursive_path_search(next_node, new_path)
                    else:
                        return
            try:
                branch = node.leaving_branches[0]
                if branch not in path:
                    path.append(branch)
                    node = self.nodes[path[-1].end_node_id]
                else:
                    return
            except IndexError:
                # current node is the end node of the signal flow graph
                return

    @property
    def paths(self) -> list[Path]:
        """
        Returns a list of all paths in the signal-flow graph.
        """
        if not self._paths: self._search_paths()
        return self._paths

    @property
    def forward_paths(self) -> list[Path]:
        """
        Returns a list of all forward paths in the signal-flow graph.
        """
        if not self._paths: self._search_paths()
        if self._forward_paths is None:
            self._forward_paths = []
            for path in self._paths:
                fwd_path = Path()
                for branch in path:
                    if branch.feedback:
                        break
                    elif branch.end_node_id != self.end_node.name:
                        fwd_path.append(branch)
                    else:
                        fwd_path.append(branch)
                        if fwd_path not in self._forward_paths:
                            self._forward_paths.append(fwd_path)
                        break
        return self._forward_paths

    @property
    def loops(self) -> list[Loop]:
        """
        Returns a list of all loops in the signal-flow graph.
        """
        if not self._paths: self._search_paths()
        if self._loops is None:
            self._loops = LoopList()
            for path in self._paths:
                reversed_path = path[-1::-1]
                i = 1
                try:
                    while reversed_path[i].start_node_id != path[-1].end_node_id:
                        i += 1
                    loop = reversed_path[:i + 1]
                    loop = Loop(loop[-1::-1])
                    if loop not in self._loops:
                        self._loops.append(loop)
                except IndexError:
                    # path is not a loop
                    continue
        return self._loops

    @staticmethod
    def _are_the_same(group_i: list[Loop], group_j: list[Loop]) -> bool:
        """
        Returns True if `group_i` and `group_j` contain the same loops, else
        False is returned.
        """
        flags = [False] * len(group_i)
        for i in range(len(group_i)):
            loop_i = group_i[i]
            for j in range(len(group_j)):
                loop_j = group_j[j]
                if loop_j is loop_i:
                    flags[i] = True
                    break
        if all(flags):
            return True
        return False

    def non_touching_loop_groups(
        self,
        loops: list[Loop] | None = None
    ) -> list[list[Loop]]:
        """
        This method accepts a list of loops (if `loops` is None, all the loops
        in the signal-flow graph will be considered). Within this list, it
        searches for groups of loops that are non-touching (i.e., loops that
        have no nodes in common).

        Returns
        -------
        A list with groups of non-touching loops (i.e., a list of lists).
        """
        if loops is None: loops = self.loops
        non_touching_loop_groups = []
        # create a list of non-touching loop groups:
        for i in range(len(loops)):
            loop_i = loops[i]
            group_i = [loop_i]
            for j in range(len(loops)):
                if j != i:
                    loop_j = loops[j]
                    if loop_j.is_touching(loop_i):
                        continue
                    else:
                        # all loops in a non-touching loop group must be
                        # non-touching:
                        for loop in group_i[1:]:
                            if loop.is_touching(loop_j):
                                break
                        else:
                            group_i.append(loop_j)
            if len(group_i) > 1:
                non_touching_loop_groups.append(group_i)
        # remove redundant groups from the list of non-touching loop groups:
        for group_i in non_touching_loop_groups[:-1]:
            for group_j in non_touching_loop_groups[1:]:
                if self._are_the_same(group_i, group_j):
                    j = non_touching_loop_groups.index(group_j)
                    non_touching_loop_groups.pop(j)
        return non_touching_loop_groups

    def non_touching_loop_combinations(
        self,
        size: int,
        loops: list[Loop] | None = None
    ) -> list[tuple[Loop, ...]]:
        """
        This method builds upon the method `non_touching_loop_groups`.
        Within each group of non-touching loops, multiple combinations of a
        certain size may be possible.
        The method returns a list with all the possible combinations having the
        size specified by parameter `size`.
        """
        _non_touching_loop_combs = []
        for group in self.non_touching_loop_groups(loops):
            combs = list(itertools.combinations(group, size))
            if combs: _non_touching_loop_combs.extend(combs)
        # Remove any redundant combinations:
        non_touching_loop_combs = []
        for comb in _non_touching_loop_combs:
            if comb not in non_touching_loop_combs:
                non_touching_loop_combs.append(comb)
        return non_touching_loop_combs

    def touching_loops(self, fwd_path: Path) -> list[Loop]:
        """
        Returns a list of all the loops that touch the given forward path.
        """
        touching_loops = []
        for loop in self.loops:
            if loop.is_touching(fwd_path):
                touching_loops.append(loop)
        return touching_loops

    def forward_path_gains(self) -> list[TTransferFunction]:
        """
        Returns a list with the gain of each forward path in the graph.
        """
        fwd_path_gains = [path.gain for path in self.forward_paths]
        return fwd_path_gains

    def loop_gains(
        self,
        loops: list[Loop] | None = None
    ) -> list[TTransferFunction]:
        """
        Returns a list with the gain of each loop in list `loops`. If `loops`
        is None, all loops in the graph are taken.
        """
        if loops is None: loops = self.loops
        loop_gains = [loop.gain for loop in loops]
        return loop_gains

    def non_touching_loop_combination_gains(
        self,
        loops: list[Loop] | None = None
    ) -> dict[int, list[TTransferFunction]]:
        """
        This method builds upon the method `non_touching_loop_combinations`.
        After in each group of non-touching loops all the possible combinations
        of a certain size are determined, the loop gain of each combination is
        calculated.
        The method returns a dictionary of which the keys refer to a combination
        size and each key is associated to a list of which the elements are the
        loop gains of each combination of that size.
        """
        if loops is None: loops = self.loops
        comb_gain_dict = {}
        size = 2
        while True:
            comb_list = self.non_touching_loop_combinations(size, loops)
            if comb_list:
                comb_gain_list = []
                for comb in comb_list:
                    comb_gain = comb[0].gain
                    for loop in comb[1:]:
                        comb_gain *= loop.gain
                    comb_gain_list.append(comb_gain)
                comb_gain_dict[size] = comb_gain_list
                size += 1
            else:
                break
        return comb_gain_dict

    def summed_loop_gains(
        self,
        loops: list[Loop] | None = None
    ) -> int | TTransferFunction:
        """
        Returns the sum of the loop gains.
        """
        if loops is None: loops = self.loops
        return sum(self.loop_gains(loops))

    def summed_non_touching_loop_combination_gains(
        self,
        loops: list[Loop] | None = None
    ) -> dict[int, TTransferFunction]:
        """
        This method builds upon the method `get_non_touching_loop_combination_gains`.
        For each combination size, the loop gains in the combination are summed.
        """
        if loops is None: loops = self.loops
        d = {
            size: sum(loop_gains)
            for size, loop_gains
            in self.non_touching_loop_combination_gains(loops).items()
        }
        # noinspection PyTypeChecker
        return d

    @property
    def denominator(self) -> TTransferFunction:
        """
        Returns the denominator of the resulting transfer function.
        """
        if self._check_if_transfer_function():
            delta = TransferFunction(1) - self.summed_loop_gains()
        else:
            delta = sp.Integer(1) - self.summed_loop_gains()
        for size, gain in self.summed_non_touching_loop_combination_gains().items():
            if size % 2 == 0:
                delta += gain
            else:
                delta -= gain
        return delta

    def delta_forward_path(self, fwd_path: Path) -> TTransferFunction:
        """
        Get the delta of the given forward path. Calculation is the same as
        for `denominator` but without the loops that touch the given forward
        path.
        """
        touching_loops = self.touching_loops(fwd_path)
        loops = [loop for loop in self.loops if loop not in touching_loops]
        if self._check_if_transfer_function():
            delta = TransferFunction(1) - self.summed_loop_gains(loops)
        else:
            delta = sp.Integer(1) - self.summed_loop_gains(loops)
        for size, gain in self.summed_non_touching_loop_combination_gains(loops).items():
            if size % 2 == 0:
                delta += gain
            else:
                delta -= gain
        return delta

    @property
    def numerator(self) -> int | TTransferFunction:
        """
        Returns the numerator of the resulting transfer function.
        """
        num = sum(
            p.gain * self.delta_forward_path(p)
            for p in self.forward_paths
        )
        return num

    @property
    def transfer_function(self) -> TTransferFunction:
        """
        Returns the transfer function of the signal-flow graph using Mason's
        rule.
        """
        G = self.numerator / self.denominator
        return G
