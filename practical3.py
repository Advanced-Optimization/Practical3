import glob
import os
import sys
import traceback

import numpy as np
import Sofa
import Sofa.ImGui as MyGui

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")


def _resolve_tray_mesh_path():
    """Resolve tray mesh across local lab copies and emio-labs installs."""
    scene_dir = os.path.dirname(os.path.realpath(__file__))

    direct_candidates = [
        os.path.normpath(os.path.join(scene_dir, "../../data/meshes/tray.stl")),
        os.path.normpath(os.path.join(scene_dir, "../data/meshes/tray.stl")),
        os.path.normpath(os.path.join(scene_dir, "data/meshes/tray.stl")),
    ]

    # Try parent roots to support running from copied lab folders.
    probe = scene_dir
    for _ in range(8):
        direct_candidates.append(os.path.join(probe, "data/meshes/tray.stl"))
        direct_candidates.append(os.path.join(probe, "assets/data/meshes/tray.stl"))
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    # Also try emio-labs versioned installs under the user home folder.
    emio_candidates = sorted(
        glob.glob(os.path.expanduser("~/emio-labs/*/assets/data/meshes/tray.stl"))
    )
    direct_candidates.extend(emio_candidates)

    for path in direct_candidates:
        if os.path.isfile(path):
            return path

    return None


def _vec3_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _default_task_tuning():
    """Return one selected parameter set for the task."""

    # ============= Exercise 1: Fill in below =================
    # Starting from fail_params, tune the parameters until your
    # project works in simulation
    success_params = {
    }
    # =========================================================

    # ============= Exercise 2: Fill in below =================
    # Starting from success_params, make any required adjustments
    # until your code runs successfully on the real robot.
    real_params = {
    }
    # =========================================================

    fail_params = {
        "gripper_opening_open": 30.0,
        "gripper_opening_closed": 12.0,
        "gripper_opening_min": 5.0,
        "gripper_opening_max": 35.0,
        "pick_position": [5.0, -165.0, 20.0],
        "place_position": [-24.0, -165.0, -12.0],
        "object_position": [-3.0, -170.0, 12.0],
        "hover_lift_height": 42.0,
        "pick_height_offset": 9.0,
        "place_height_offset": 11.0,
        "lift_success_delta": 30.0,
    }
    return fail_params


class PickAndPlaceEvaluator(Sofa.Core.Controller):

    def __init__(
        self,
        root,
        block_mo,
        tcp_mo,
        gripper_mo,
        gripper_opening,
        object_position,
        pick_position,
        place_position,
        gripper_opening_closed,
        lift_success_delta,
    ):
        Sofa.Core.Controller.__init__(self)
        self.name = "PickAndPlaceEvaluator"

        self.root = root
        self.block_mo = block_mo
        self.tcp_mo = tcp_mo
        self.gripper_mo = gripper_mo
        self.gripper_opening = gripper_opening
        self.object_position = np.array(object_position)
        self.pick_position = np.array(pick_position)
        self.place_position = np.array(place_position)
        self.gripper_opening_closed = float(gripper_opening_closed)
        self.tray_height = float(object_position[1])
        self.lift_success_delta = float(lift_success_delta)
        self.attach_distance_threshold = 11.0
        self.attach_opening_threshold = self.gripper_opening_closed + 0.5
        self.attach_offset = np.array([0.0, -10.0, 0.0], dtype=float)
        self.is_attached = False
        self.lifted = False
        self.placed = False
        self.is_falling = False
        self._runtime_error_logged = False
        self._frame_count = 0
        print("[P3][Evaluator] Initialized")

    def _to_flat_array(self, data):
        if hasattr(data, "value"):
            data = data.value
        return np.array(data, dtype=float).reshape(-1)

    def _tcp_position(self):
        values = self._to_flat_array(self.tcp_mo.position)
        if values.size < 3:
            return None
        return values[0:3]

    def _block_position(self):
        values = self._to_flat_array(self.block_mo.position)
        if values.size < 3:
            return None
        return values[0:3]

    def _set_block_position(self, xyz):
        values = self._to_flat_array(self.block_mo.position)
        if values.size >= 7:
            q = values[3:7]
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        self.block_mo.position.value = [list(np.array(xyz, dtype=float)) + list(q)]

    def _get_gripper_opening(self):
        opening_data = self.gripper_opening
        if hasattr(opening_data, "value"):
            opening_data = opening_data.value
        opening_array = np.array(opening_data, dtype=float).reshape(-1)
        if opening_array.size == 0:
            return 0.0
        # restLengths can be multi-valued; median is more stable than index 0.
        return float(np.median(opening_array))

    def _gripper_end_positions(self):
        values = np.array(
            self.gripper_mo.position.value,
            dtype=float,
        ).reshape(-1, 3)
        if values.shape[0] < 2:
            return None, None
        return values[0], values[1]

    def _grasp_point(self):
        end_a, end_b = self._gripper_end_positions()
        if end_a is None or end_b is None:
            return None, None
        return 0.5 * (end_a + end_b), _vec3_distance(end_a, end_b)

    def _attach_point(self, tcp_pos, grasp_point):
        """Prefer TCP-based pickup anchor; fall back to grasp midpoint."""
        if tcp_pos is not None:
            return np.array(tcp_pos, dtype=float) + self.attach_offset
        return np.array(grasp_point, dtype=float)

    def onAnimateBeginEvent(self, e):
        _ = e
        try:
            self._frame_count += 1
            elapsed = float(self.root.time.value)
            self.root.taskElapsed.value = elapsed

            tcp_pos = self._tcp_position()
            block_pos = self._block_position()
            if tcp_pos is None or block_pos is None:
                print(
                    "[P3][Evaluator] Invalid positions",
                    "frame=",
                    self._frame_count,
                    "tcp=",
                    tcp_pos,
                    "block=",
                    block_pos,
                )
                return

            opening = self._get_gripper_opening()
            grasp_point, end_distance = self._grasp_point()
            if grasp_point is None:
                print("[P3][Evaluator] Missing gripper end positions")
                return
            attach_point = self._attach_point(tcp_pos, grasp_point)
            dist_mid = _vec3_distance(grasp_point, block_pos)
            dist_attach = _vec3_distance(attach_point, block_pos)
            self.root.tcpX.value = float(tcp_pos[0])
            self.root.tcpY.value = float(tcp_pos[1])
            self.root.tcpZ.value = float(tcp_pos[2])
            self.root.blockX.value = float(block_pos[0])
            self.root.blockY.value = float(block_pos[1])
            self.root.blockZ.value = float(block_pos[2])
            self.root.graspMidX.value = float(grasp_point[0])
            self.root.graspMidY.value = float(grasp_point[1])
            self.root.graspMidZ.value = float(grasp_point[2])
            self.root.distToMidpoint.value = float(dist_mid)
            self.root.distToAttach.value = float(dist_attach)
            self.root.gripperOpeningForGrasp.value = float(opening)

            if (not self.is_attached) and (not self.placed):
                close_enough = dist_attach <= self.attach_distance_threshold
                closed_enough = opening <= self.attach_opening_threshold

                if close_enough and closed_enough:
                    self.is_attached = True
                    self.is_falling = False
                    print(
                        "[P3][Evaluator] ATTACH frame",
                        self._frame_count,
                        "opening=",
                        round(opening, 3),
                        "dist_attach=",
                        round(dist_attach, 3),
                    )

            if self.is_attached:
                self._set_block_position(attach_point)
                block_pos = self._block_position()
                if block_pos is None:
                    return
                if block_pos[1] > self.object_position[1] + self.lift_success_delta:
                    self.lifted = True
                    print("[P3][Evaluator] LIFTED frame", self._frame_count)

                if opening > self.gripper_opening_closed:
                    self.is_attached = False
                    self.is_falling = True
                    print("[P3][Evaluator] PLACED frame", self._frame_count)

            elif self.is_falling:
                fall_y = max(self.tray_height, block_pos[1] - 4.0)
                self._set_block_position([block_pos[0], fall_y, block_pos[2]])
                if fall_y <= self.tray_height:
                    self.is_falling = False
                    self.placed = True

            score = 0.0
            if self.lifted:
                score += 50.0
            if self.placed:
                score += 50.0

            if self.placed:
                score += max(0.0, 20.0 - elapsed)

            self.root.taskScore.value = float(score)
            self.root.taskLifted.value = bool(self.lifted)
            self.root.taskPlaced.value = bool(self.placed)
        except Exception as exc:
            if not self._runtime_error_logged:
                Sofa.msg_error(
                    "PickAndPlaceEvaluator",
                    "Runtime error in evaluator: " + str(exc),
                )
                print("[P3][Evaluator] EXCEPTION", repr(exc))
                traceback.print_exc()
                self._runtime_error_logged = True


class AutoPickAndPlaceDemo(Sofa.Core.Controller):

    def __init__(
        self,
        root,
        target_mo,
        block_mo,
        gripper_opening,
        pick_position,
        place_position,
        gripper_opening_open,
        gripper_opening_closed,
        gripper_opening_min,
        gripper_opening_max,
        hover_lift_height,
        pick_height_offset,
        place_height_offset,
    ):
        Sofa.Core.Controller.__init__(self)
        self.name = "AutoPickAndPlaceDemo"
        self.root = root
        self.target_mo = target_mo
        self.block_mo = block_mo
        self.gripper_opening = gripper_opening
        self.pick_position = np.array(pick_position, dtype=float)
        self.place_position = np.array(place_position, dtype=float)
        self.opening_open = float(gripper_opening_open)
        self.opening_closed = float(gripper_opening_closed)
        self.opening_min = float(gripper_opening_min)
        self.opening_max = float(gripper_opening_max)
        self.hover_lift_height = float(hover_lift_height)
        self.pick_height_offset = float(pick_height_offset)
        self.place_height_offset = float(place_height_offset)
        # mm/s limit for stable beam deformation while opening/closing
        self.opening_rate = 45.0
        self.demo_duration = 16.0
        self.is_active = True
        self.start_time = None
        self.last_phase = None
        print("[P3][Demo] Initialized")

    def _set_target_position(self, xyz):
        self.target_mo.position.value = [
            [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        ]

    def _set_gripper_opening(self, value):
        target = float(np.clip(value, self.opening_min, self.opening_max))
        self.root.commandedOpening.value = target

        opening_data = self.gripper_opening

        try:
            if hasattr(opening_data, "value"):
                opening_data.value = target
            elif isinstance(opening_data, (list, tuple)):
                # Array-like but immutable, cannot set
                print(
                    "[P3][Demo] WARNING: gripper_opening is immutable "
                    f"(type={type(opening_data).__name__})"
                )
                return
            else:
                # Try slice assignment for array-like objects
                opening_data[:] = target
        except (TypeError, AttributeError, IndexError) as e:
            # Silent fail - gripper_opening may be in an inconsistent state
            print(
                f"[P3][Demo] WARNING: faild to set gripper: {e} "
                f"(type={type(opening_data).__name__})"
            )

    def _set_block_position(self, xyz):
        values = self.block_mo.position.value[0]
        if len(values) >= 7:
            q = values[3:7]
        else:
            q = [0.0, 0.0, 0.0, 1.0]
        self.block_mo.position.value = [
            [
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                q[0],
                q[1],
                q[2],
                q[3],
            ]
        ]

    def onAnimateBeginEvent(self, e):
        _ = e
        if not self.is_active:
            return

        current_time = float(self.root.time.value)
        if self.start_time is None:
            self.start_time = current_time

        elapsed = current_time - self.start_time

        approach_pick = self.pick_position + np.array(
            [0.0, self.hover_lift_height, 0.0]
        )
        at_pick = self.pick_position + np.array([0.0, self.pick_height_offset, 0.0])
        lift_pick = self.pick_position + np.array([0.0, self.hover_lift_height, 0.0])
        approach_place = self.place_position + np.array(
            [0.0, self.hover_lift_height, 0.0]
        )
        at_place = self.place_position + np.array([0.0, self.place_height_offset, 0.0])

        if elapsed < 2.0:
            phase = 0
            self._set_target_position(approach_pick)
            self._set_gripper_opening(self.opening_open)
        elif elapsed < 4.0:
            phase = 1
            self._set_target_position(at_pick)
            self._set_gripper_opening(self.opening_open)
        elif elapsed < 6.0:
            phase = 2
            self._set_target_position(at_pick)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 8.0:
            phase = 3
            self._set_target_position(lift_pick)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 10.0:
            phase = 4
            self._set_target_position(approach_place)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 12.0:
            phase = 5
            self._set_target_position(at_place)
            self._set_gripper_opening(self.opening_closed)
        elif elapsed < 14.0:
            phase = 6
            self._set_target_position(at_place)
            self._set_gripper_opening(self.opening_open)
        elif elapsed < self.demo_duration:
            phase = 7
            self._set_target_position(approach_place)
            self._set_gripper_opening(self.opening_open)
        else:
            self.is_active = False
            self.root.autoDemoActive.value = False
            print("[P3][Demo] timeline ended; manual GUI control restored")
            return

        if phase != self.last_phase:
            self.last_phase = phase

        self.root.targetX.value = float(self.target_mo.position.value[0][0])
        self.root.targetY.value = float(self.target_mo.position.value[0][1])
        self.root.targetZ.value = float(self.target_mo.position.value[0][2])


def createScene(rootnode):
    from parts.controllers.assemblycontroller import AssemblyController
    from parts.emio import Emio, getParserArgs
    from parts.gripper import Gripper
    from utils.header import addHeader, addSolvers

    print("[P3][Scene] createScene start")
    args = getParserArgs()
    tuning = _default_task_tuning()

    settings, modelling, simulation = addHeader(
        rootnode,
        inverse=True,
        withCollision=True,
    )
    print("[P3][Scene] header added")
    settings.addObject(
        "RequiredPlugin",
        name="collision_geometry",
        pluginName=["Sofa.Component.Collision.Geometry"],
    )
    addSolvers(simulation)
    print("[P3][Scene] solvers added")

    rootnode.dt = 0.05
    rootnode.gravity = [0.0, -9810.0, 0.0]
    rootnode.VisualStyle.displayFlags.value = ["hideBehavior", "hideWireframe"]

    rootnode.addData(name="taskScore", type="float", value=0.0)
    rootnode.addData(name="taskElapsed", type="float", value=0.0)
    rootnode.addData(name="taskLifted", type="bool", value=False)
    rootnode.addData(name="taskPlaced", type="bool", value=False)
    rootnode.addData(name="autoDemoActive", type="bool", value=True)
    rootnode.addData(name="targetX", type="float", value=0.0)
    rootnode.addData(name="targetY", type="float", value=0.0)
    rootnode.addData(name="targetZ", type="float", value=0.0)
    rootnode.addData(name="tcpX", type="float", value=0.0)
    rootnode.addData(name="tcpY", type="float", value=0.0)
    rootnode.addData(name="tcpZ", type="float", value=0.0)
    rootnode.addData(name="blockX", type="float", value=0.0)
    rootnode.addData(name="blockY", type="float", value=0.0)
    rootnode.addData(name="blockZ", type="float", value=0.0)
    rootnode.addData(name="graspMidX", type="float", value=0.0)
    rootnode.addData(name="graspMidY", type="float", value=0.0)
    rootnode.addData(name="graspMidZ", type="float", value=0.0)
    rootnode.addData(name="distToMidpoint", type="float", value=0.0)
    rootnode.addData(name="distToAttach", type="float", value=0.0)
    rootnode.addData(name="gripperOpeningForGrasp", type="float", value=0.0)
    rootnode.addData(name="commandedOpening", type="float", value=0.0)
    print("[P3][Scene] task data added")

    emio = Emio(
        name="Emio",
        legsName=["blueleg"],
        legsModel=["beam"],
        legsPositionOnMotor=[
            "counterclockwisedown",
            "clockwisedown",
            "counterclockwisedown",
            "clockwisedown",
        ],
        centerPartName="whitepart",
        centerPartType="deformable",
        centerPartModel="beam",
        centerPartClass=Gripper,
        platformLevel=2,
        extended=True,
    )
    if not emio.isValid():
        print("[P3][Scene] emio invalid")
        return
    print("[P3][Scene] emio valid")

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    emio.addObject(AssemblyController(emio))
    print("[P3][Scene] emio inserted")

    # Workaround: skip gripper collision setup here because the current
    # gripper collision helper builds a filename with a DataString + str,
    # which crashes scene loading in this environment.

    tray_mesh = _resolve_tray_mesh_path()
    if tray_mesh is not None:
        tray = modelling.addChild("Tray")
        tray.addObject(
            "MeshSTLLoader",
            filename=tray_mesh,
            translation=[0, 20, 0],
        )
        tray.addObject(
            "OglModel",
            src=tray.MeshSTLLoader.linkpath,
            color=[0.3, 0.3, 0.3, 0.2],
        )
        print("[P3][Scene] tray added from", tray_mesh)
    else:
        print("[P3][Scene] WARNING: tray.stl not found; skipping tray visual")

    emio.effector.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[0, 0, 0, 0, 0, 0, 1] * 4,
    )
    emio.effector.addObject("RigidMapping", rigidIndexPerPoint=[0, 1, 2, 3])
    print("[P3][Scene] effector added")

    effector_target = modelling.addChild("Target")
    effector_target.addObject("EulerImplicitSolver", firstOrder=True)
    effector_target.addObject(
        "CGLinearSolver",
        iterations=50,
        tolerance=1e-10,
        threshold=1e-10,
    )
    target_mo = effector_target.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[0, -150, 0, 0, 0, 0, 1],
        showObject=False,
        showObjectScale=20,
    )
    print("[P3][Scene] target added")

    emio.addInverseComponentAndGUI(
        effector_target.getMechanicalState().position.linkpath,
        barycentric=True,
    )
    tcp = modelling.addChild("TCP")
    tcp_mo = tcp.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=emio.effector.EffectorCoord.barycenter.linkpath,
    )
    MyGui.setIPController(
        rootnode.Modelling.Target,
        tcp,
        rootnode.ConstraintSolver,
    )
    print("[P3][Scene] inverse + TCP added")

    opening_data = emio.centerpart.Effector.Distance.DistanceMapping.restLengths
    gripper_state = emio.centerpart.Effector.getMechanicalState()
    print("[P3][Scene] opening data bound")
    MyGui.MoveWindow.addAccessory(
        "Gripper's opening (mm)",
        opening_data,
        tuning["gripper_opening_min"],
        tuning["gripper_opening_max"],
    )
    MyGui.ProgramWindow.addGripper(
        opening_data,
        tuning["gripper_opening_min"],
        tuning["gripper_opening_max"],
    )
    MyGui.ProgramWindow.importProgram(
        os.path.dirname(__file__) + "/mypickandplace.crprog"
    )
    MyGui.IOWindow.addSubscribableData("/Gripper", opening_data)
    print("[P3][Scene] GUI controls added")

    # User-tuned waypoints and object spawn pose.
    pick_position = tuning["pick_position"]
    place_position = tuning["place_position"]
    object_position = tuning["object_position"]

    # Keep the block kinematic for robust task evaluation.
    block = modelling.addChild("Block")
    block_mo = block.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=[
            object_position[0],
            object_position[1],
            object_position[2],
            0,
            0,
            0,
            1,
        ],
        showObject=False,
    )

    block_visual = block.addChild("Visual")
    block_visual.addObject(
        "MechanicalObject",
        template="Rigid3",
        position=block_mo.position.linkpath,
    )
    block_visual.addObject(
        "MeshOBJLoader",
        name="loader",
        filename=os.path.dirname(__file__) + "/cube.obj",
        translation=[-5.0, -32.0, 5.0],
    )
    block_visual.addObject("MeshTopology", src="@loader")
    block_visual.addObject("OglModel", color=[0.82, 0.62, 0.25, 1.0])
    block_visual.addObject("RigidMapping", index=0)

    print("[P3][Scene] kinematic block added")

    pick_marker = modelling.addChild("PickMarker")
    pick_marker.addObject(
        "MechanicalObject",
        template="Vec3",
        position=[pick_position],
        showObject=False,
        showObjectScale=8,
        showColor=[1.0, 0.3, 0.3, 1.0],
    )
    place_marker = modelling.addChild("PlaceMarker")
    place_marker.addObject(
        "MechanicalObject",
        template="Vec3",
        position=[place_position],
        showObject=False,
        showObjectScale=8,
        showColor=[0.3, 1.0, 0.3, 1.0],
    )
    print("[P3][Scene] markers added")

    rootnode.addObject(
        PickAndPlaceEvaluator(
            root=rootnode,
            block_mo=block_mo,
            tcp_mo=tcp_mo,
            gripper_mo=gripper_state,
            gripper_opening=opening_data,
            object_position=object_position,
            pick_position=pick_position,
            place_position=place_position,
            gripper_opening_closed=tuning["gripper_opening_closed"],
            lift_success_delta=tuning["lift_success_delta"],
        )
    )
    print("[P3][Scene] evaluator added")

    rootnode.addObject(
        AutoPickAndPlaceDemo(
            root=rootnode,
            target_mo=target_mo,
            block_mo=block_mo,
            gripper_opening=opening_data,
            pick_position=pick_position,
            place_position=place_position,
            gripper_opening_open=tuning["gripper_opening_open"],
            gripper_opening_closed=tuning["gripper_opening_closed"],
            gripper_opening_min=tuning["gripper_opening_min"],
            gripper_opening_max=tuning["gripper_opening_max"],
            hover_lift_height=tuning["hover_lift_height"],
            pick_height_offset=tuning["pick_height_offset"],
            place_height_offset=tuning["place_height_offset"],
        )
    )
    print("[P3][Scene] demo path added")

    MyGui.MyRobotWindow.addInformation("Task score", rootnode.taskScore)
    MyGui.MyRobotWindow.addInformation("Task lifted", rootnode.taskLifted)
    MyGui.MyRobotWindow.addInformation("Task placed", rootnode.taskPlaced)
    MyGui.MyRobotWindow.addInformation(
        "Auto demo active",
        rootnode.autoDemoActive,
    )
    MyGui.MyRobotWindow.addInformation(
        "Task elapsed (s)",
        rootnode.taskElapsed,
    )
    MyGui.PlottingWindow.addData("tcp x", rootnode.tcpX)
    MyGui.PlottingWindow.addData("tcp y", rootnode.tcpY)
    MyGui.PlottingWindow.addData("tcp z", rootnode.tcpZ)
    MyGui.PlottingWindow.addData("block x", rootnode.blockX)
    MyGui.PlottingWindow.addData("block y", rootnode.blockY)
    MyGui.PlottingWindow.addData("block z", rootnode.blockZ)
    MyGui.PlottingWindow.addData("grasp midpoint x", rootnode.graspMidX)
    MyGui.PlottingWindow.addData("grasp midpoint y", rootnode.graspMidY)
    MyGui.PlottingWindow.addData("grasp midpoint z", rootnode.graspMidZ)
    MyGui.PlottingWindow.addData(
        "diag distance block-midpoint",
        rootnode.distToMidpoint,
    )
    MyGui.PlottingWindow.addData(
        "diag distance block-attach",
        rootnode.distToAttach,
    )
    MyGui.PlottingWindow.addData(
        "diag gripper opening (GUI)",
        rootnode.gripperOpeningForGrasp,
    )
    MyGui.PlottingWindow.addData(
        "diag gripper opening command",
        rootnode.commandedOpening,
    )
    print("[P3][Scene] info fields added")

    if args.connection:
        emio.addConnectionComponents()
        print("[P3][Scene] connection components added")

    print("[P3][Scene] createScene complete")
    return rootnode
