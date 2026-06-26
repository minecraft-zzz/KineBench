TASK_PROMPTS: dict[str, str] = {
    "StackCube-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StackCube-v1. There are two small cubes on the tabletop: one red cube and one green cube. At the start, keep the gripper open. Move the gripper to directly above the green cube, lower it slightly, then close the gripper to grasp the green cube. Lift the green cube upward, move it to directly above the red cube, then open the gripper to place the green cube on top of the red cube. Finally, slightly lift the gripper upward to complete the task.",
    "PullCube-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: PullCube-v1. A small blue cube is on the tabletop, along with a red-and-white target marker. Close the gripper and move it downward in front of the blue cube, then move straight backward to drag the cube to the target marker without lifting it.",
    "CloseFaucet-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseFaucet-v1. Move the gripper to the faucet handle and rotate it clockwise until the handle reaches the closed orientation.",
    "OpenFaucet-v1": "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenFaucet-v1. Move the gripper to the faucet handle and rotate it counter-clockwise until the handle reaches the open orientation.",
}

VISUAL_LOCK_PREFIX = (
    "Use the given first frame as the exact visual reference. The robotic arm must keep the exact same appearance "
    "as in the first frame. Preserve its geometry, proportions, materials, colors, surface details, and structure. "
    "The robotic arm is rigid and non-deformable. Only physically correct joint rotations and translations are allowed. "
    "The camera is static and locked off. The composition remains identical to the first frame. "
)


def prompt_for_task(env_id: str, prompts: dict[str, str] | None = None, add_visual_lock: bool = True) -> str:
    table = dict(TASK_PROMPTS)
    if prompts:
        table.update(prompts)
    if env_id not in table:
        raise KeyError(f"No prompt configured for env_id={env_id}")
    return (VISUAL_LOCK_PREFIX if add_visual_lock else "") + table[env_id]

