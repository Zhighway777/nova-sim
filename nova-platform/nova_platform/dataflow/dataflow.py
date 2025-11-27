from enum import Enum
from queue import PriorityQueue
from typing import Generator, List, Set, Tuple
import networkx as nx
import logging

from nova_platform.base_model import DataflowActionMemoryStat
from nova_platform.cost_service.compute.base_compute_model import DataflowAction
from nova_platform.executor.nova_platform_barrier import BARRIER
from nova_platform.executor.nova_platform_event import BossaNovaEvent


logger = logging.getLogger(__name__)


class Dataflow():
    #构建action的图依赖关系，将list转换为action的dag
    def _build_dag(self, action_list: List[DataflowAction]):
        dag = nx.DiGraph()
        roots = []
        action_map = {}
        for action in action_list:
            action_id = action.get_action_id()
            action_map[action_id] = action
            if not action.get_parent_ids():
                roots.append(action_id)
                dag.add_node(action_id)
            if action.get_child_ids():
                for child_action_id in action.get_child_ids():
                    dag.add_edge(action_id, child_action_id)

        self.dag = dag
        self._action_map = action_map
        self._roots = roots
        self.queue = PriorityQueue()

    def execute_dataflow(self) -> Generator[DataflowAction | BARRIER, Tuple[float, bool, DataflowAction | BARRIER], None]:
        visited = set()

        # queue = self._roots.copy()
        for action in self._roots:
            self.queue.put((0, action))
        # bfs
        while True:
            if not self.queue.empty():
                priority, action_id = self.queue.get()
                if action_id in visited:
                    continue
                if not all(p in visited for p in self.dag.predecessors(action_id)):
                    # queue.put((priority, action_id))
                    # TODO: need review
                    continue
                # logger.debug('Executing:%s', self._action_map[vertex])
                next_ref, is_done, stat = yield action_id, self._action_map[action_id]
            else:
                next_ref, is_done, stat = yield None, None

            if isinstance(stat, BARRIER) or isinstance(stat, BossaNovaEvent):
                pass
            elif not is_done:
                # if action is not done, put it back to queue
                self.queue.put((next_ref, action_id))
            elif is_done:
                # if action is done, mark it as visited and add its children to queue
                visited.add(action_id)
                children = self.dag.successors(action_id)
                # queue.extend(children)
                for _action_id in children:
                    if (next_ref, _action_id) not in self.queue.queue:
                        self.queue.put((next_ref, _action_id))
                    else:
                        pass
            elif (next_ref, is_done, stat) == (None, None, None):
                break
            else:
                raise RuntimeError
                # if (next_ref, is_done, stat) == (None, None, None):
                #     break
        # check visited no of actions match dag size
        assert (len(visited) == len(self.dag.nodes),
                "visited node size doesn't match dag node size")
