TASK_PROMPTS: dict[str, str] = {
    'CloseBox-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseBox-v1. A box is placed on its side in front of the robot arm. The box lid is open. Move the gripper to hover above the lid, then apply a gentle downward press with a slight rightward component to fold the lid shut until it is fully closed. Finish when the lid is completely closed, then retract the gripper.',
    'CloseDrawer-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseDrawer-v1. A cabinet/drawer unit is in front of the robot arm, and one drawer is open. Move the gripper forward and slightly downward to reach the inside/back side of the open drawer front. Then push the gripper forward to slide the drawer inward until it is fully closed and flush. Stop once the drawer is completely closed, then retract the gripper upward.',
    'CloseFaucet-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseFaucet-v1. A faucet is in front of the robot arm. The faucet model may vary across multiple variants (shape/size may change), so focus on the handle orientation. The faucet starts in an open state with the handle pointing to the left. Move the gripper down and slightly forward to approach the front/right side of the handle, then sweep the gripper to the right and backward to rotate the handle clockwise until the handle points backward (closed position). Maintain controlled contact and stop when the handle reaches the fully closed orientation. Do not rotate counter-clockwise (that would open it more); do not push the handle in the wrong direction; do not ignore the handle orientation when faucet geometry changes.',
    'CloseLaptopEasy-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseLaptopEasy-v1. A laptop is in front of the robot arm. The laptop lid is open only slightly (small opening angle). Move the gripper forward toward the front/top edge of the lid, then pull the lid backward toward the robot while pressing downward, closing the lid fully in one smooth continuous motion. Because the lid is only slightly open, the forward reach should be short and the closing can be done in a single combined pull-and-press action. Stop once the laptop is completely closed, then retract the gripper upward. Do not try to open the laptop; do not push the lid further backward to increase the opening; do not grasp and lift the laptop.',
    'CloseLaptopHard-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: CloseLaptopHard-v1. A laptop is in front of the robot arm. The laptop lid is open widely (large opening angle). Reach the gripper forward to the front/top region of the lid. First pull the lid backward toward the robot to reduce the opening angle. After the lid is partially closed, press downward to close it completely. Because the lid starts widely open, the reach should be larger and the closing typically requires two phases: pull back, then press down. Stop once the laptop is fully closed, then retract the gripper upward. Do not try to open the laptop further; do not press straight down too early when the lid is still widely open; do not push from behind the lid in a way that increases the opening.',
    'LiftPegUpright-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: LiftPegUpright-v1. A tabletop single-arm robot with a parallel gripper faces a two-colored peg. Lower the gripper to the tip/end of the red section, align the gripper with the peg, close the gripper to grasp the red part, and lift/rotate the peg into an upright standing pose so that the red section points upward. Then move the gripper down and open the gripper to release the peg so it stands on the table by itself.',
    'OpenBoxEasy-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxEasy-v1. A box is in front of the robot arm. The lid is closed but already slightly open. Move the gripper to the front of the lid, then push the lid forward in one smooth motion until the box is fully open. Stop when the lid is fully open, then retract the gripper.',
    'OpenBoxEasy-v2': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxEasy-v2. A box with a partially open lid is placed on a tabletop, and the box is oriented toward the right. Move the gripper to the edge of the box lid. Rotate the gripper to face the box, then push the lid upward and forward until the box is fully open.',
    'OpenBoxHard-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxHard-v1. A box is in front of the robot arm. The lid is fully closed and hard to open. Reach the gripper forward while rotating the gripper so that the gripper tip can approach the side edge of the lid. Use the gripper tip to pry the lid up. Once the lid is lifted to a workable angle, push forward to open the lid fully.',
    'OpenBoxHard-v2': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenBoxHard-v2. A box is positioned in front of the robot arm, and the box lid is tightly closed. Move the gripper downward to the front edge of the lid. First, rotate the gripper upward to slightly lift/pry the lid and create a usable opening angle. After the lid has been lifted to a small angle, continue pushing forward and upward until the box is fully open.',
    'OpenDrawer-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenDrawer-v1. A drawer unit is placed sideways in front of the robot arm, and the drawer is slightly open. Rotate the gripper 90 degrees around the vertical (z) axis to align the gripper tip with the drawer gap. Move the gripper down so the tip inserts into the opened gap. Then pull outward along the drawer opening direction to open the drawer fully. Finish when the drawer is clearly open, then retract the gripper. Do not push inward to close the drawer.',
    'OpenFaucet-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenFaucet-v1. A faucet is in front of the robot arm. The faucet may appear in multiple variants (shape/size/style can change), so focus on the handle orientation rather than the exact geometry. The faucet starts closed, with the handle pointing inward (toward the inside of the scene). Move the gripper down to the right side of the handle, make controlled contact, then sweep the gripper left and slightly forward to rotate the handle counter-clockwise until the handle points to the left, indicating the faucet is open. Stop once the handle reaches the fully open orientation.do not rotate clockwise (that would close it more); Do not push the handle in the wrong direction when the faucet model changes; do not ignore the handle orientation.',
    'OpenLaptopEasy-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenLaptopEasy-v1. A laptop is in front of the robot arm. The laptop lid is almost closed but already slightly ajar (the opening angle is not tiny). Move the gripper down to the front edge of the lid, then push the lid forward and upward in a smooth motion until the laptop is fully open. Because the lid is already partially open, this task can typically be completed by a single continuous forward-up push. Stop when the lid reaches the fully open position, then retract the gripper. Do not press downward to close the lid; do not try to pry or lift aggressively; do not push in a direction that makes the lid close further.',
    'OpenLaptopHard-v1': "In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: OpenLaptopHard-v1. A laptop is in front of the robot arm. The laptop lid is tightly closed (very small or near-zero opening angle). Move the gripper down to the front edge of the lid. First, rotate the gripper upward to lift/pry the lid up slightly and create a usable opening angle. After the lid is lifted to a small angle, push forward and upward until the laptop is fully open. This task usually requires two phases: pry up first, then push open. Stop when the lid is fully open, then retract the gripper. Do not only push forward when the lid is fully shut (it won't open); do not press downward; do not attempt to close the laptop.",
    'PickFruits-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: PickFruits-v1. The target fruit is the {fruit_name}. Move the gripper to the {fruit_name}, grasp it securely, lift it from the table, and place it into the target container while avoiding the other fruits.',
    'PullCube-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: PullCube-v1. A small blue cube is on the tabletop, along with a red-and-white target marker. Close the gripper and move it downward in front of the blue cube, then move straight backward to drag the cube to the target marker without lifting it.',
    'StackCube-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StackCube-v1. There are two small cubes on the tabletop: one red cube and one green cube. At the start, keep the gripper open. Move the gripper to directly above the green cube, lower it slightly, then close the gripper to grasp the green cube. Lift the green cube upward, move it to directly above the red cube, then open the gripper to place the green cube on top of the red cube. Finally, slightly lift the gripper upward to complete the task.',
    'StoreCube-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StoreCube-v1. A cabinet/drawer unit is positioned in front of the robot arm, with one drawer open, and a small red cube located above the drawer. Move the gripper toward the small cube and grasp it, then move the gripper above the open drawer and release the gripper so that the cube drops into the drawer. Move the gripper backward and slightly downward to reach the inside/back side of the open drawer front. Then push the gripper forward to slide the drawer inward until it is fully closed and flush with the cabinet.',
    'StoreCube-v2': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StoreCube-v2. A cabinet/drawer unit is positioned in front of the robot arm, with one drawer open, and a small red cube located above the drawer. First, lift the gripper to avoid contacting the cube, then close the gripper and move it downward in front of the red cube, and then move the gripper backward to drag the cube along the tabletop until it falls into the drawer. Next, push the drawer inward until it is fully closed.',
    'StoreFruitsBox-v1': 'In the following video, based on the current visual state, determine the task progress and continue to complete the remaining steps. Task: StoreFruitsBox-v1. Move the fruits into the box in this exact order: {fruit_sequence_text}. For each fruit, move the gripper to the fruit, grasp it securely, lift it, place it into the box, then continue with the next fruit in the specified order.',
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

PICK_FRUIT_NAMES = ["apple", "banana", "peach", "pear", "orange", "strawberry"]
STORE_FRUIT_NAMES = {0: "apple", 1: "banana", 2: "orange"}


def _flatten_ints(value) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(_flatten_ints(item))
        return out
    return [int(value)]


def _join_sequence(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def task_context_for_env(env_id: str, env) -> dict:
    unwrapped = getattr(env, "unwrapped", env)
    if env_id == "PickFruits-v1":
        fruit_ids = _flatten_ints(getattr(unwrapped, "fruit_id_to_pick"))
        fruit_id = fruit_ids[0]
        fruit_name = PICK_FRUIT_NAMES[fruit_id]
        return {"fruit_id_to_pick": fruit_id, "fruit_name": fruit_name}
    if env_id == "StoreFruitsBox-v1":
        fruit_ids = _flatten_ints(getattr(unwrapped, "fruit_to_pick"))
        fruit_names = [STORE_FRUIT_NAMES[fruit_id] for fruit_id in fruit_ids]
        return {"fruit_to_pick": fruit_ids, "fruit_sequence": fruit_names, "fruit_sequence_text": _join_sequence(fruit_names)}
    return {}


def _pick_fruits_prompt(fruit_name: str) -> str:
    return (
        "In the following video, based on the current visual state, determine the task progress and continue to "
        f"complete the remaining steps. Task: PickFruits-v1. The target fruit is the {fruit_name}. "
        f"Move the gripper to the {fruit_name}, grasp it securely, lift it from the table, and place it into the "
        "target container while avoiding the other fruits."
    )


def _store_fruits_box_prompt(fruit_names: list[str]) -> str:
    seq = _join_sequence(fruit_names)
    return (
        "In the following video, based on the current visual state, determine the task progress and continue to "
        "complete the remaining steps. Task: StoreFruitsBox-v1. Move the fruits into the box in this exact order: "
        f"{seq}. For each fruit, move the gripper to the fruit, grasp it securely, lift it, place it into the box, "
        "then continue with the next fruit in the specified order."
    )


def prompt_for_env(env_id: str, env, prompts: dict[str, str] | None = None, add_visual_lock: bool = True) -> tuple[str, dict]:
    context = task_context_for_env(env_id, env)
    if env_id not in {"PickFruits-v1", "StoreFruitsBox-v1"}:
        return prompt_for_task(env_id, prompts, add_visual_lock=add_visual_lock), context

    table = dict(TASK_PROMPTS)
    if prompts:
        table.update(prompts)
    prompt_template = table[env_id]
    prompt = prompt_template.format(**context)
    return (VISUAL_LOCK_PREFIX if add_visual_lock else "") + prompt, context

