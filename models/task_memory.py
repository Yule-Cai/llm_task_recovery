# models/task_memory.py
from envs.task_env import (
    TASK_PICK_DELIVER, TASK_PATROL, TASK_SEARCH,
    PHASE_DONE,
    ANOMALY_NONE, ANOMALY_OBSTRUCTION, ANOMALY_DISPLACEMENT, ANOMALY_INVALIDATION,
)


class TaskMemory:
    """
    Lightweight symbolic task state tracker supporting all three task types.
    Serialises into a compact string injected into the LLM recovery prompt.
    """

    INSTRUCTIONS = {
        TASK_PICK_DELIVER: "Go to the item location, pick it up, then deliver it to the goal zone.",
        TASK_PATROL:       "Visit waypoint 1, then waypoint 2, then waypoint 3 in order.",
        TASK_SEARCH:       "Explore the environment to find the item, pick it up, then return to base.",
    }

    PHASE_NAMES = {
        TASK_PICK_DELIVER: {0: "go_to_item",  1: "go_to_goal",  10: "done"},
        TASK_PATROL:       {0: "go_to_wp1",   1: "go_to_wp2",   2: "go_to_wp3", 10: "done"},
        TASK_SEARCH:       {0: "explore",      1: "go_to_item",  2: "go_to_base", 10: "done"},
    }

    def __init__(self, task_type=TASK_PICK_DELIVER):
        self.task_type   = task_type
        self.instruction = self.INSTRUCTIONS[task_type]
        self.completed   = []
        self.active      = self.PHASE_NAMES[task_type][0]
        self.anomaly_log = []

    def update_phase(self, task_phase):
        phase_map = self.PHASE_NAMES[self.task_type]
        name      = phase_map.get(task_phase, "unknown")

        if task_phase == PHASE_DONE:
            if self.active not in self.completed:
                self.completed.append(self.active)
            self.active = "done"
        else:
            prev = self.active
            self.active = name
            if prev != name and prev not in self.completed and prev != "done":
                self.completed.append(prev)

    def record_anomaly(self, anomaly_type, step):
        self.anomaly_log.append({"type": anomaly_type, "step": step})

    def to_prompt_string(self, anomaly_type=ANOMALY_NONE):
        lines = [
            "Original instruction: " + self.instruction,
            "Completed subtasks: " + (", ".join(self.completed) if self.completed else "none"),
            "Active subtask: " + self.active,
        ]
        desc = {
            ANOMALY_NONE:         "",
            ANOMALY_OBSTRUCTION:  "New obstacles have appeared, blocking the current route.",
            ANOMALY_DISPLACEMENT: "The item has moved to a new location.",
            ANOMALY_INVALIDATION: "The goal zone is now inaccessible. An alternate goal has been set.",
        }.get(anomaly_type, "")
        if desc:
            lines.append("Anomaly: " + desc)
        return "\n".join(lines)